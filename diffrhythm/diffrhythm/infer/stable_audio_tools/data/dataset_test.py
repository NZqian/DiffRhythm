import importlib
import numpy as np
import io
import os
import posixpath
import random
import re
import subprocess
import time
import torch
import torchaudio
import webdataset as wds
#import redis
import pickle

from aeiou.core import is_silence
from os import path
from pedalboard.io import AudioFile
import pedalboard
from torchaudio import transforms as T
from typing import Optional, Callable, List
import random
import threading

from .utils import Stereo, Mono, PhaseFlipper, PadCrop_Normalized_T
from .aslp.tools import LanceReader
from .aslp.data import AudioData
#from data.utils import Stereo, Mono, PhaseFlipper, PadCrop_Normalized_T
#from data.lance_utils.aslp_utils.tools import LanceReader
#from data.lance_utils.aslp_utils.data import AudioData

torch.multiprocessing.set_start_method('spawn', force=True)

AUDIO_KEYS = ("flac", "wav", "mp3", "m4a", "ogg", "opus")

# fast_scandir implementation by Scott Hawley originally in https://github.com/zqevans/audio-diffusion/blob/main/dataset/dataset.py

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def normalize_audio(y, target_dbfs=0):
    max_amplitude = torch.max(torch.abs(y))

    target_amplitude = 10.0**(target_dbfs / 20.0)
    scale_factor = target_amplitude / max_amplitude

    normalized_audio = y * scale_factor

    return normalized_audio

def fast_scandir(
    dir:str,  # top-level directory at which to begin scanning
    ext:list,  # list of allowed file extensions,
    #max_size = 1 * 1000 * 1000 * 1000 # Only files < 1 GB
    ):
    "very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243"
    subfolders, files = [], []
    ext = ['.'+x if x[0]!='.' else x for x in ext]  # add starting period to extensions if needed
    try: # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try: # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    file_ext = os.path.splitext(f.name)[1].lower()
                    is_hidden = os.path.basename(f.path).startswith(".")

                    if file_ext in ext and not is_hidden:
                        files.append(f.path)
            except:
                pass 
    except:
        pass

    for dir in list(subfolders):
        sf, f = fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files

def keyword_scandir(
    dir: str,  # top-level directory at which to begin scanning
    ext: list,  # list of allowed file extensions
    keywords: list,  # list of keywords to search for in the file name
):
    "very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243"
    subfolders, files = [], []
    # make keywords case insensitive
    keywords = [keyword.lower() for keyword in keywords]
    # add starting period to extensions if needed
    ext = ['.'+x if x[0] != '.' else x for x in ext]
    banned_words = ["paxheader", "__macosx"]
    try:  # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try:  # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    is_hidden = f.name.split("/")[-1][0] == '.'
                    has_ext = os.path.splitext(f.name)[1].lower() in ext
                    name_lower = f.name.lower()
                    has_keyword = any(
                        [keyword in name_lower for keyword in keywords])
                    has_banned = any(
                        [banned_word in name_lower for banned_word in banned_words])
                    if has_ext and has_keyword and not has_banned and not is_hidden and not os.path.basename(f.path).startswith("._"):
                        files.append(f.path)
            except:
                pass
    except:
        pass

    for dir in list(subfolders):
        sf, f = keyword_scandir(dir, ext, keywords)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files

def get_audio_filenames(
    paths: list,  # directories in which to search
    keywords=None,
    exts=['.wav']
    #exts=['.wav', '.mp3', '.flac', '.ogg', '.aif', '.opus']
):
    "recursively get a list of audio filenames"
    filenames = []
    if type(paths) is str:
        paths = [paths]
    for path in paths:               # get a list of relevant filenames
        if keywords is not None:
            subfolders, files = keyword_scandir(path, exts, keywords)
        else:
            subfolders, files = fast_scandir(path, exts)
        filenames.extend(files)
    return filenames

class LocalDatasetConfig:
    def __init__(
        self,
        id: str,
        path: str,
        custom_metadata_fn: Optional[Callable[[str], str]] = None
    ):
        self.id = id
        self.path = path
        self.custom_metadata_fn = custom_metadata_fn

class SampleDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        configs,
        sample_size=65536, 
        sample_rate=48000, 
        keywords=None, 
        random_crop=True,
        force_channels="stereo"
    ):
        super().__init__()
        self.filenames = []
        
        self.sample_size=sample_size

        self.augs = torch.nn.Sequential(
            PhaseFlipper(),
        )

        self.root_paths = []

        self.pad_crop = PadCrop_Normalized_T(sample_size, sample_rate, randomize=random_crop)

        self.force_channels = force_channels

        self.encoding = torch.nn.Sequential(
            Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
            Mono() if self.force_channels == "mono" else torch.nn.Identity(),
        )

        self.sr = sample_rate

        self.custom_metadata_fns = {}

        self.reader = LanceReader(configs[0].path, target_cls=AudioData)
        self.offset = 0
        self.update_ids()

        self._cache_ref_count = 0
        self.n_cache_reuse = 1
        self.cache_audio = None

        #for config in configs:
        #    self.root_paths.append(config.path)
        #    self.filenames.extend(get_audio_filenames(config.path, keywords))
        #    if config.custom_metadata_fn is not None:
        #        self.custom_metadata_fns[config.path] = config.custom_metadata_fn

        print(f'Found {len(self.filenames)} files')
    def update_ids(self):
        self.filenames = self.reader.get_ids()



    def load_file(self, filename):
        ext = filename.split(".")[-1]

        if ext == "mp3":
            with AudioFile(filename) as f:
                audio = f.read(f.frames)
                audio = torch.from_numpy(audio)
                in_sr = f.samplerate
        else:
            #start_time = time.time()
            audio, in_sr = torchaudio.load(filename, format=ext)
            #end_time = time.time()
            #print("load audio", end_time - start_time, in_sr, self.sr)

        if in_sr != self.sr:
            resample_tf = T.Resample(in_sr, self.sr)
            audio = resample_tf(audio)

        return audio

    def __len__(self):
        return len(self.filenames)
        #return self.cache_size

    def get_item_redis(self, idx):
        idx = idx % self.cache_size
        r = self.redis_pool
        audio = r.get(str(idx))
        if audio:
            audio = torch.from_numpy(pickle.loads(audio))
        else:
            i = 0
            while True:
                if i == 10:
                    raise
                rand_idx = random.randint(0, self.cache_size - 1) 
                audio = r.get(str(rand_idx))
                if audio:
                    audio = torch.from_numpy(pickle.loads(audio))
                    break
                i += 1
        #self.cache_audio = audio
        return audio

    def get_item_disk(self, idx):
        print("loading")
        audio = self.load_file(self.filenames[idx])
        print("loaded")
        self.cache_audio = audio
        self.cache_thread_running = False

    def get_item_lance(self, idx):
        audio = self.reader.get_datas_by_rows([idx])[0].audio
        audio_length = audio.shape[-1]
        audio = np.stack((audio[:audio_length // 2], audio[audio_length // 2:]), axis=0)
        audio = torch.from_numpy(audio).float() / 32768.
        return audio

    def __getitem__(self, idx):
        self.offset += 1
        if self.offset == self.__len__():
            self.offset = 0
        #idx = idx % self.cache_size
        #idx = 0
        audio_filename = self.filenames[idx]
        try:
            start_time = time.time()

            audio = self.get_item_lance(idx)
            assert list(audio.shape) == [2, 441000], f"{audio.shape}"
            if is_silence(audio):
                return torch.zeros(2, self.sample_size), torch.zeros(2, self.sample_size), None
            audio = audio.clamp(-1, 1)
            audio = normalize_audio(audio, -6)
            stft = torch.stft(audio, n_fft=2048, hop_length=512, win_length=2048, return_complex=True)
            stft = torch.abs(stft)
            stft = torch.mean(stft, 0)
            stft = spectral_normalize_torch(stft)

            if stft[900:, :].mean() < -3.3:
                raise

            #audio = None
            #for i in range(3):
            #    audio = self.get_item_lance(idx)
            #    audio = normalize_audio(audio, -6)
            #    stft = torch.stft(audio, n_fft=2048, hop_length=512, win_length=2048, return_complex=True)
            #    stft = torch.abs(stft)
            #    if torch.abs(stft[:, 800:, :]).mean() < 0.04:
            #        audio = None
            #        idx = random.randint(0, self.__len__()-1)
            #    else:
            #        break
            #if audio is None:
            #    raise

            #print("idx", idx)
            #audio = self.reader.get_datas_by_rowids([audio_filename._rowid])[0].audio
            #if self._cache_ref_count == 0:
            #    audio = self.reader.get_datas_by_rows([idx])[0].audio
            #    audio_length = audio.shape[-1]
            #    audio = np.stack((audio[:audio_length // 2], audio[audio_length // 2:]), axis=0)
            #    audio = torch.from_numpy(audio).float() / 32768.
            #    self.cache_audio = audio
            #else:
            #    audio = self.cache_audio
            #    self._cache_ref_count -= 1
            #print(audio.dtype, audio.min(), audio.max())

            #if self.cache_audio is None:
            #    #self.get_item_redis(idx)
            #    self.get_item_disk(idx)
            #audio = self.cache_audio
            ##if random.randint(1, 10) > 8:
            #if not self.cache_thread_running:
            #    thread = threading.Thread(target=self.get_item_disk, args=(idx,))
            #    thread.start()
            #    self.cache_thread_running = True


            #if self._cache_ref_count == 0:
            #    #audio = self.load_file(audio_filename)
            #    audio = self.get_item_redis(idx)
            #    self.cache_audio = audio
            #    self._cache_ref_count = self.n_cache_reuse
            #else:
            #    audio = self.cache_audio
            #    self._cache_ref_count -= 1
            
            #print(audio.shape)
            #audio = self.load_file(audio_filename)
            #audio = self.r.get(str(idx))
            #if audio:
            #    audio = torch.from_numpy(pickle.loads(audio))
            #    load_type = "redis"
            #else:
            #    load_idx = random.randint(0, len(self.filenames)-1)
            #    audio_filename = self.filenames[load_idx]
            #    audio = self.load_file(audio_filename)
            #    self.r.set(str(idx), pickle.dumps(audio.numpy()))
            #    expire_time = random.randint(1800 * 1, 1800 * 2)
            #    self.r.expire(str(idx), expire_time)
            #    load_type = "disk"
            #audio = torch.rand(2, 441000)
                
            audio, t_start, t_end, seconds_start, seconds_total, padding_mask = self.pad_crop(audio)

            # Run augmentations on this sample (including random crop)
            if self.augs is not None:
                audio = self.augs(audio)

            audio = audio.clamp(-1, 1)

            # Encode the file to assist in prediction
            if self.encoding is not None:
                audio = self.encoding(audio)

            info = {}

            info["path"] = audio_filename

            for root_path in self.root_paths:
                if root_path in audio_filename:
                    info["relpath"] = path.relpath(audio_filename, root_path)

            info["timestamps"] = (t_start, t_end)
            info["seconds_start"] = seconds_start
            info["seconds_total"] = seconds_total
            info["padding_mask"] = padding_mask

            end_time = time.time()

            info["load_time"] = end_time - start_time
            #print("load_time", end_time - start_time)

            for custom_md_path in self.custom_metadata_fns.keys():
                if custom_md_path in audio_filename:
                    custom_metadata_fn = self.custom_metadata_fns[custom_md_path]
                    custom_metadata = custom_metadata_fn(info, audio)
                    info.update(custom_metadata)

                if "__reject__" in info and info["__reject__"]:
                    return self[random.randrange(len(self))]

            aug_ratio = random.random() * 21 #[0, 30]
            if aug_ratio > 14:
                aug_audio = audio
            else:
                aug_audio = pedalboard.MP3Compressor(aug_ratio / 2)(audio.numpy(), self.sr)
                aug_audio = torch.from_numpy(aug_audio)
            aug_audio = normalize_audio(aug_audio, -6)

            assert not torch.any(torch.isnan(audio))
            assert not torch.any(torch.isinf(audio))
            assert not torch.any(torch.isnan(aug_audio))
            assert not torch.any(torch.isinf(aug_audio))

            return (audio, aug_audio, info)
        except Exception as e:
            raise
            #print(f'Couldn\'t load file {audio_filename}: {e}')
            # return self[random.randrange(len(self))]

def group_by_keys(data, keys=wds.tariterators.base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.
    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if wds.tariterators.trace:
            print(
                prefix,
                suffix,
                current_sample.keys() if isinstance(current_sample, dict) else None,
            )
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        if current_sample is None or prefix != current_sample["__key__"]:
            if wds.tariterators.valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffix in current_sample:
            print(f"{fname}: duplicate file name in tar file {suffix} {current_sample.keys()}")
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if wds.tariterators.valid_sample(current_sample):
        yield current_sample

wds.tariterators.group_by_keys = group_by_keys

# S3 code and WDS preprocessing code based on implementation by Scott Hawley originally in https://github.com/zqevans/audio-diffusion/blob/main/dataset/dataset.py

def get_s3_contents(dataset_path, s3_url_prefix=None, filter='', recursive=True, debug=False, profile=None):
    """
    Returns a list of full S3 paths to files in a given S3 bucket and directory path.
    """
    # Ensure dataset_path ends with a trailing slash
    if dataset_path != '' and not dataset_path.endswith('/'):
        dataset_path += '/'
    # Use posixpath to construct the S3 URL path
    bucket_path = posixpath.join(s3_url_prefix or '', dataset_path)
    # Construct the `aws s3 ls` command
    cmd = ['aws', 's3', 'ls', bucket_path]

    if profile is not None:
        cmd.extend(['--profile', profile])

    if recursive:
        # Add the --recursive flag if requested
        cmd.append('--recursive')
    
    # Run the `aws s3 ls` command and capture the output
    run_ls = subprocess.run(cmd, capture_output=True, check=True)
    # Split the output into lines and strip whitespace from each line
    contents = run_ls.stdout.decode('utf-8').split('\n')
    contents = [x.strip() for x in contents if x]
    # Remove the timestamp from lines that begin with a timestamp
    contents = [re.sub(r'^\S+\s+\S+\s+\d+\s+', '', x)
                if re.match(r'^\S+\s+\S+\s+\d+\s+', x) else x for x in contents]
    # Construct a full S3 path for each file in the contents list
    contents = [posixpath.join(s3_url_prefix or '', x)
                for x in contents if not x.endswith('/')]
    # Apply the filter, if specified
    if filter:
        contents = [x for x in contents if filter in x]
    # Remove redundant directory names in the S3 URL
    if recursive:
        # Get the main directory name from the S3 URL
        main_dir = "/".join(bucket_path.split('/')[3:])
        # Remove the redundant directory names from each file path
        contents = [x.replace(f'{main_dir}', '').replace(
            '//', '/') for x in contents]
    # Print debugging information, if requested
    if debug:
        print("contents = \n", contents)
    # Return the list of S3 paths to files
    return contents


def get_all_s3_urls(
    names=[],           # list of all valid [LAION AudioDataset] dataset names
    # list of subsets you want from those datasets, e.g. ['train','valid']
    subsets=[''],
    s3_url_prefix=None,  # prefix for those dataset names
    recursive=True,     # recursively list all tar files in all subdirs
    filter_str='tar',   # only grab files with this substring
    # print debugging info -- note: info displayed likely to change at dev's whims
    debug=False,
    profiles={},        # dictionary of profiles for each item in names, e.g. {'dataset1': 'profile1', 'dataset2': 'profile2'}
):
    "get urls of shards (tar files) for multiple datasets in one s3 bucket"
    urls = []
    for name in names:
        # If s3_url_prefix is not specified, assume the full S3 path is included in each element of the names list
        if s3_url_prefix is None:
            contents_str = name
        else:
            # Construct the S3 path using the s3_url_prefix and the current name value
            contents_str = posixpath.join(s3_url_prefix, name)
        if debug:
            print(f"get_all_s3_urls: {contents_str}:")
        for subset in subsets:
            subset_str = posixpath.join(contents_str, subset)
            if debug:
                print(f"subset_str = {subset_str}")
            # Get the list of tar files in the current subset directory
            profile = profiles.get(name, None)
            tar_list = get_s3_contents(
                subset_str, s3_url_prefix=None, recursive=recursive, filter=filter_str, debug=debug, profile=profile)
            for tar in tar_list:
                # Escape spaces and parentheses in the tar filename for use in the shell command
                tar = tar.replace(" ", "\ ").replace(
                    "(", "\(").replace(")", "\)")
                # Construct the S3 path to the current tar file
                s3_path = posixpath.join(name, subset, tar) + " -"
                # Construct the AWS CLI command to download the current tar file
                if s3_url_prefix is None:
                    request_str = f"pipe:aws s3 --cli-connect-timeout 0 cp {s3_path}"
                else:
                    request_str = f"pipe:aws s3 --cli-connect-timeout 0 cp {posixpath.join(s3_url_prefix, s3_path)}"
                if profiles.get(name):
                    request_str += f" --profile {profiles.get(name)}"
                if debug:
                    print("request_str = ", request_str)
                # Add the constructed URL to the list of URLs
                urls.append(request_str)
    return urls


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    print(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def is_valid_sample(sample):
    has_json = "json" in sample
    has_audio = "audio" in sample
    is_silent = is_silence(sample["audio"])
    is_rejected = "__reject__" in sample["json"] and sample["json"]["__reject__"]

    return has_json and has_audio and not is_silent and not is_rejected

class S3DatasetConfig:
    def __init__(
        self,
        id: str,
        s3_path: str,
        custom_metadata_fn: Optional[Callable[[str], str]] = None,
        profile: Optional[str] = None,
    ):
        self.id = id
        self.path = s3_path
        self.custom_metadata_fn = custom_metadata_fn
        self.profile = profile
        self.urls = []

    def load_data_urls(self):
        self.urls = get_all_s3_urls(
            names=[self.path],
            s3_url_prefix=None,
            recursive=True,
            profiles={self.path: self.profile} if self.profile else {},
        )

        return self.urls

class LocalWebDatasetConfig:
    def __init__(
        self,
        id: str,
        path: str,
        custom_metadata_fn: Optional[Callable[[str], str]] = None,
        profile: Optional[str] = None,
    ):
        self.id = id
        self.path = path
        self.custom_metadata_fn = custom_metadata_fn
        self.urls = []

    def load_data_urls(self):

        self.urls = fast_scandir(self.path, ["tar"])[1]

        return self.urls

def audio_decoder(key, value):
    # Get file extension from key
    ext = key.split(".")[-1]

    if ext in AUDIO_KEYS:
        return torchaudio.load(io.BytesIO(value))
    else:
        return None

def collation_fn(samples):
        batched = list(zip(*samples))
        result = []
        for b in batched:
            if isinstance(b[0], (int, float)):
                b = np.array(b)
            elif isinstance(b[0], torch.Tensor):
                b = torch.stack(b)
            elif isinstance(b[0], np.ndarray):
                b = np.array(b)
            else:
                b = b
            result.append(b)
        return result

class WebDatasetDataLoader():
    def __init__(
        self,
        datasets: List[S3DatasetConfig],
        batch_size,
        sample_size,
        sample_rate=48000,
        num_workers=8,
        epoch_steps=1000,
        random_crop=True,
        force_channels="stereo",
        augment_phase=True,
        **data_loader_kwargs
    ):

        self.datasets = datasets

        self.sample_size = sample_size
        self.sample_rate = sample_rate
        self.random_crop = random_crop
        self.force_channels = force_channels
        self.augment_phase = augment_phase

        urls = [dataset.load_data_urls() for dataset in datasets]

        # Flatten the list of lists of URLs
        urls = [url for dataset_urls in urls for url in dataset_urls]

        # Shuffle the urls
        random.shuffle(urls)

        self.dataset = wds.DataPipeline(
            wds.ResampledShards(urls),
            wds.tarfile_to_samples(handler=log_and_continue),
            wds.decode(audio_decoder, handler=log_and_continue),
            wds.map(self.wds_preprocess, handler=log_and_continue),
            wds.select(is_valid_sample),
            wds.to_tuple("audio", "json", handler=log_and_continue),
            #wds.shuffle(bufsize=1000, initial=5000),
            wds.batched(batch_size, partial=False, collation_fn=collation_fn),
        ).with_epoch(epoch_steps//num_workers if num_workers > 0 else epoch_steps)

        self.data_loader = wds.WebLoader(self.dataset, num_workers=num_workers, **data_loader_kwargs)

    def wds_preprocess(self, sample):

        found_key, rewrite_key = '', ''
        for k, v in sample.items():  # print the all entries in dict
            for akey in AUDIO_KEYS:
                if k.endswith(akey):
                    # to rename long/weird key with its simpler counterpart
                    found_key, rewrite_key = k, akey
                    break
            if '' != found_key:
                break
        if '' == found_key:  # got no audio!
            return None  # try returning None to tell WebDataset to skip this one

        audio, in_sr = sample[found_key]
        if in_sr != self.sample_rate:
            resample_tf = T.Resample(in_sr, self.sample_rate)
            audio = resample_tf(audio)

        if self.sample_size is not None:
            # Pad/crop and get the relative timestamp
            pad_crop = PadCrop_Normalized_T(
                self.sample_size, randomize=self.random_crop, sample_rate=self.sample_rate)
            audio, t_start, t_end, seconds_start, seconds_total, padding_mask = pad_crop(
                audio)
            sample["json"]["seconds_start"] = seconds_start
            sample["json"]["seconds_total"] = seconds_total
            sample["json"]["padding_mask"] = padding_mask
        else:
            t_start, t_end = 0, 1

        # Check if audio is length zero, initialize to a single zero if so
        if audio.shape[-1] == 0:
            audio = torch.zeros(1, 1)

        # Make the audio stereo and augment by randomly inverting phase
        augs = torch.nn.Sequential(
            Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
            Mono() if self.force_channels == "mono" else torch.nn.Identity(),
            PhaseFlipper() if self.augment_phase else torch.nn.Identity()
        )

        audio = augs(audio)

        sample["json"]["timestamps"] = (t_start, t_end)

        if "text" in sample["json"]:
            sample["json"]["prompt"] = sample["json"]["text"]

        # Check for custom metadata functions
        for dataset in self.datasets:
            if dataset.custom_metadata_fn is None:
                continue
        
            if dataset.path in sample["__url__"]:
                custom_metadata = dataset.custom_metadata_fn(sample["json"], audio)
                sample["json"].update(custom_metadata)

        if found_key != rewrite_key:   # rename long/weird key with its simpler counterpart
            del sample[found_key]

        sample["audio"] = audio

        # Add audio to the metadata as well for conditioning
        sample["json"]["audio"] = audio
        
        return sample

def create_dataloader_from_config(dataset_config, batch_size, sample_size, sample_rate, audio_channels=2, num_workers=4):

    dataset_type = dataset_config.get("dataset_type", None)

    assert dataset_type is not None, "Dataset type must be specified in dataset config"

    if audio_channels == 1:
        force_channels = "mono"
    else:
        force_channels = "stereo"

    if dataset_type == "audio_dir":

        audio_dir_configs = dataset_config.get("datasets", None)

        assert audio_dir_configs is not None, "Directory configuration must be specified in datasets[\"dataset\"]"

        configs = []

        for audio_dir_config in audio_dir_configs:
            audio_dir_path = audio_dir_config.get("path", None)
            assert audio_dir_path is not None, "Path must be set for local audio directory configuration"

            custom_metadata_fn = None
            custom_metadata_module_path = audio_dir_config.get("custom_metadata_module", None)

            if custom_metadata_module_path is not None:
                spec = importlib.util.spec_from_file_location("metadata_module", custom_metadata_module_path)
                metadata_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(metadata_module)                

                custom_metadata_fn = metadata_module.get_custom_metadata

            configs.append(
                LocalDatasetConfig(
                    id=audio_dir_config["id"],
                    path=audio_dir_path,
                    custom_metadata_fn=custom_metadata_fn
                )
            )

        train_set = SampleDataset(
            configs,
            sample_rate=sample_rate,
            sample_size=sample_size,
            random_crop=dataset_config.get("random_crop", True),
            force_channels=force_channels
        )

        return torch.utils.data.DataLoader(train_set, batch_size, shuffle=True,
                                num_workers=num_workers, persistent_workers=True, pin_memory=True, drop_last=True, collate_fn=collation_fn)

    elif dataset_type in ["s3", "wds"]: # Support "s3" type for backwards compatibility
        wds_configs = []

        for wds_config in dataset_config["datasets"]:

            custom_metadata_fn = None
            custom_metadata_module_path = wds_config.get("custom_metadata_module", None)

            if custom_metadata_module_path is not None:
                spec = importlib.util.spec_from_file_location("metadata_module", custom_metadata_module_path)
                metadata_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(metadata_module)                

                custom_metadata_fn = metadata_module.get_custom_metadata

            if "s3_path" in wds_config:

                wds_configs.append(
                    S3DatasetConfig(
                        id=wds_config["id"],
                        s3_path=wds_config["s3_path"],
                        custom_metadata_fn=custom_metadata_fn,
                        profile=wds_config.get("profile", None),
                    )
                )
            
            elif "path" in wds_config:
                    
                    wds_configs.append(
                        LocalWebDatasetConfig(
                            id=wds_config["id"],
                            path=wds_config["path"],
                            custom_metadata_fn=custom_metadata_fn
                        )
                    )

        return WebDatasetDataLoader(
            wds_configs,
            sample_rate=sample_rate,
            sample_size=sample_size,
            batch_size=batch_size,
            random_crop=dataset_config.get("random_crop", True),
            num_workers=num_workers,
            persistent_workers=True,
            force_channels=force_channels,
            epoch_steps=dataset_config.get("epoch_steps", 2000)
        ).data_loader

if __name__ == '__main__':
    from tqdm import tqdm
    import json
    import soundfile as sf
    import time
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    dataset_config_path = 'configs/dataset_configs/local_training_lance.json'
    with open(dataset_config_path) as f:
        dataset_config = json.load(f)
    train_dl = create_dataloader_from_config(
        dataset_config, 
        batch_size=32, 
        num_workers=4,
        sample_rate=44100,
        sample_size=12288,
        audio_channels=2,
    )

    for i, data in tqdm(enumerate(train_dl)):
        print(i, data[0].min(), data[0].max(), data[0].mean(), data[0].std(), data[1].min(), data[1].max(), data[1].mean(), data[1].std())
        #audio = data[0][0].numpy().T
        #print(audio.shape)
        #sf.write("stereo.wav", data[0][0].numpy().T, 44100)
        #break
        pass

    #lance_dir = "/home/node44_tmpdata/netease/wav_lances/sq_slice"

    #reader = LanceReader(lance_dir, target_cls=AudioData)
    #ids = reader.get_ids()
    #print(len(ids))
    #rows = [i for i in range(len(ids))]
    ##import pdb;pdb.set_trace()
    #for _ in range(10):
    #    time_start = time.time()
    #    rows = [i for i in range(len(ids))]
    #    random_wav:AudioData = reader.get_datas_by_rows(random.sample(rows, 1))[0]
    #    print(time.time() - time_start)
    #out_path = 'test.wav'

    #wav_data = random_wav.audio
    #print(type(wav_data), wav_data.shape)
    #wav_stereo = np.stack((wav_data[:441000], wav_data[441000:]), axis=1)
    ##wav_stereo = np.reshape(wav_data, (2,-1))
    #print(wav_stereo.shape)
    ## audio = (audio * numpy.iinfo(numpy.int16).max).astype(numpy.int16)
    ## import pdb;pdb.set_trace()
    ## wav_data = wav_data / np.iinfo(wav_data.dtype).max

    #sf.write("mono.wav", wav_data, random_wav.sample_rate)
    #sf.write("stereo.wav", wav_stereo, random_wav.sample_rate)