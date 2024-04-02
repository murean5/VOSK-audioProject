import copy
import json
import random

from dataclass_wizard import Container, JSONListWizard, JSONFileWizard
import math
import wave
from dataclasses import dataclass
from random import randint, choice, shuffle
from typing import Tuple, List

import librosa
import numpy as np
import soundfile
import vosk
from librosa.core import audio
from tqdm import tqdm
from vosk import KaldiRecognizer, Model
import os

MODEL_PATH = 'vosk-model-en-us-0.22-lgraph'
SAMPLE_RATE = 22050
PATH_TO_ORIGS = 'data/global/origs'


class RecognizerWrapper:
    chunk_size = 4000
    sample_rate = SAMPLE_RATE

    def __init__(self, model_directory: str):
        self.model_directory = model_directory
        self.temporary_audio_path = 'data/tmp/tmp.wav'
        self.model = Model(model_directory)

    def _process_sample(self):
        recognizer = KaldiRecognizer(self.model, self.sample_rate)
        recognizer.SetWords(True)

        audio = wave.open(self.temporary_audio_path, 'rb')

        results = []
        while True:
            data = audio.readframes(self.chunk_size)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                part_result = json.loads(recognizer.Result())
                results.append(part_result)

        part_result = json.loads(recognizer.FinalResult())
        results.append(part_result)

        return results

    def get_transcription(self, audio_path: str):
        detected_words, time_steps = [], []

        audio_sample, _ = librosa.load(audio_path, sr=self.sample_rate)
        soundfile.write(
            file=self.temporary_audio_path,
            data=audio_sample,
            samplerate=self.sample_rate,
        )

        results = self._process_sample()

        for utterance_result in results:
            if 'result' not in utterance_result:
                continue
            for word_result in utterance_result['result']:
                if not len(word_result['word']):
                    continue

                word = word_result['word']
                start = word_result['start']
                end = word_result['end']

                detected_words.append(word)
                time_steps.append((start, end))

        return {
            "words": detected_words,
            "time_steps": time_steps,
        }


def test_recognition(audio_path, subtitles_path):
    rec_wrapper = RecognizerWrapper(MODEL_PATH)
    result = rec_wrapper.get_transcription(audio_path)
    zipped_recognition = list(result['words'])
    orig_text = ''
    clean_text = ''
    with open(subtitles_path, encoding='utf-8') as file:
        lines = file.readlines()
    for line in lines:
        orig_text += line.lower().strip() + ' '

    for symbol in orig_text:
        if symbol.isalpha() or symbol.isspace() or symbol == "'":
            clean_text += symbol
    clean_text = clean_text.split()
    correct_words = 0
    words_set = set(clean_text)
    for word in words_set:
        if clean_text.count(word) == zipped_recognition.count(word):
            correct_words += 1
    print(f'Accuracy = {round(correct_words / len(words_set) * 100, 2)}%')


global_cur_determined_words = []


def determine_words(audio_path: str, zipped=False):
    global global_cur_determined_words
    words_dict = dict()

    rec_wrapper = RecognizerWrapper(MODEL_PATH)
    result = rec_wrapper.get_transcription(audio_path)
    zipped_recognition = list(zip(result['words'], result['time_steps']))
    if zipped:
        return zipped_recognition
    global_cur_determined_words = zipped_recognition
    for word, span in zipped_recognition:
        word = word.lower()
        if word not in words_dict:
            words_dict[word] = []

        words_dict[word].append(span)
    return words_dict


def get_pair_to_swap(wd, key) -> (Tuple, Tuple):
    assert len(wd[key]) >= 2

    first = choice(wd[key])
    second = first
    while second == first:
        second = choice(wd[key])
    first = (int(first[0] * SAMPLE_RATE), int(first[1] * SAMPLE_RATE))
    second = (int(second[0] * SAMPLE_RATE), int(second[1] * SAMPLE_RATE))
    return (first, second) if first[0] < second[0] else (second, first)


global_audio_id = 0
global_log = []
global_duration_coefficients = []


@dataclass(frozen=True)
class AudioData(JSONListWizard, JSONFileWizard):
    my_path: str
    orig_path: str
    splices: List


classes_to_log = Container[AudioData]()


def span_duration(span: tuple):
    return span[1] - span[0]


global_wd = dict()
global_wd_list = []
global_bucket_id = 0


def aggressive_merging():
    global global_wd, global_wd_list, global_bucket_id

    for bucket_id in range(global_bucket_id):
        cur_orig_data, sr = librosa.load(path=f'data/global/buckets/bucket{bucket_id}.wav', sr=SAMPLE_RATE)
        cur_orig_data_list = list(cur_orig_data)
        new_data = copy.deepcopy(cur_orig_data_list)
        cur_splices = []

        for cur_word, spans in global_wd_list[bucket_id].items():
            if not (cur_word in global_wd.keys() and len(global_wd[cur_word]) >= 2):
                continue

            for cur_span in spans:

                cur_dur = cur_span[1] - cur_span[0]

                *other_spans, = filter(lambda span_: span_[-1] != bucket_id, global_wd[cur_word])

                if len(other_spans) >= 1:
                    swap_choice_span = choice(other_spans)
                    # print(len(other_spans), cur_word, swap_choice_span)
                    swap_choice_data, sr = librosa.load(path=f'{PATH_TO_ORIGS}/orig{swap_choice_span[-1]}.wav',
                                                        sr=SAMPLE_RATE)
                    swap_choice_dur = swap_choice_span[1] - swap_choice_span[0]

                    swap_word_data = list(swap_choice_data[int(swap_choice_span[0] * SAMPLE_RATE):
                                                           int(swap_choice_span[1] * SAMPLE_RATE)])

                    offset = abs(cur_dur - swap_choice_dur) / 2

                    if cur_dur >= swap_choice_dur:
                        new_data[int(cur_span[0] * SAMPLE_RATE):
                                 int(cur_span[1] * SAMPLE_RATE)] = [0] * int(offset * SAMPLE_RATE) + \
                                                                   swap_word_data + \
                                                                   [0] * int(offset * SAMPLE_RATE)
                        cur_splices.append(int(cur_span[0] * SAMPLE_RATE))
                        cur_splices.append(int((cur_span[0] + swap_choice_dur + offset * 2) * SAMPLE_RATE))

        orig_len = len(cur_orig_data_list)
        aggressive_len = len(new_data)

        if orig_len < aggressive_len:
            cur_orig_data_list += [0] * (aggressive_len - orig_len)
        else:
            new_data += [0] * (orig_len - aggressive_len)

        new_orig_path = f'{PATH_TO_ORIGS}/orig{bucket_id}.wav'
        soundfile.write(file=new_orig_path,
                        data=cur_orig_data_list,
                        samplerate=SAMPLE_RATE)
        aggressive_path = f'data/global/aggressive/aggressive{bucket_id}.wav'
        soundfile.write(file=aggressive_path,
                        data=new_data,
                        samplerate=SAMPLE_RATE)
        classes_to_log.append(AudioData(my_path=aggressive_path,
                                        orig_path=new_orig_path,
                                        splices=sorted(cur_splices)))


def create_global_dict(loaded_data, wd):
    global global_audio_id, global_bucket_id, global_wd, global_wd_list

    global_wd_list.append(wd)

    MIN_WORD_DUR = 0.15
    MIN_WORD_LEN = 3

    for word in wd.keys():
        if word not in global_wd:
            global_wd[word] = []
        global_wd[word] = global_wd[word] + list(map(lambda span: (span[0], span[1], global_bucket_id),
                                                     filter(lambda span: len(word) >= MIN_WORD_LEN and span_duration(
                                                         span) >= MIN_WORD_DUR,
                                                            wd[word])))
    soundfile.write(file=f'{PATH_TO_ORIGS}/orig{global_bucket_id}.wav',
                    data=loaded_data,
                    samplerate=SAMPLE_RATE)
    global_bucket_id += 1


def shuffle_swap(loaded_data,
                 orig_path,
                 wd,
                 words_to_add=60,
                 output_files=2,
                 left=None,
                 right=None,
                 long_trio=False,
                 long=True,
                 offset=0):
    global global_audio_id, global_log, classes_to_log

    for _ in range(output_files):
        new_data = []
        wd_copy = copy.deepcopy(wd)

        tmp = wd_copy[left:right]

        shuffle(tmp)
        wd_copy[left:right] = tmp
        words = wd_copy[:words_to_add]

        cur_path = f'data/global/shuffled/shuffle{global_audio_id}.wav'
        cur_splices = []
        cur_time = 0
        min_word_time = 0.3
        for word in words:
            span = int(word[1][0] * SAMPLE_RATE), int(word[1][1] * SAMPLE_RATE)
            dur = span[1] - span[0]
            if dur / SAMPLE_RATE < min_word_time:
                continue

            cur_splices.append(cur_time + dur)
            cur_time += dur

            new_data.extend(loaded_data[span[0]:span[1]])

        soundfile.write(file=cur_path,
                        data=new_data,
                        samplerate=int(SAMPLE_RATE))
        global_audio_id += 1

        if long_trio:
            pick_long_trio(loaded_data=new_data,
                           orig_path=cur_path,
                           wd=determine_words(cur_path, zipped=True),
                           mid_word_volume_factor=1,
                           offset=offset,
                           min_audio_time=0.3)
        else:
            classes_to_log.append(AudioData(cur_path, orig_path, cur_splices))


def pick_long_trio(loaded_data,
                   orig_path,
                   wd: list[tuple[str, tuple[float, float]]],
                   min_audio_time=0.5,
                   offset=1,
                   mid_word_volume_factor=1.0
                   ):
    global global_audio_id, global_log, classes_to_log

    for i in range(len(wd)):
        if i - 1 >= 0 and i + 1 < len(wd):
            left = wd[i - 1]
            left_word, left_span = left[0], left[1]
            left_dur = left_span[1] - left_span[0]

            mid = wd[i]
            mid_word, mid_span = mid[0], mid[1]
            mid_dur = mid_span[1] - mid_span[0]

            right = wd[i + 1]
            right_word, right_span = right[0], right[1]
            right_dur = right_span[1] - right_span[0]

            if (mid_dur >= min_audio_time and
                    left_dur >= min_audio_time and
                    right_dur >= min_audio_time):
                cur_data = loaded_data[int(max(0, (left_span[0] - offset) * SAMPLE_RATE)):
                                       int(min(len(loaded_data), (right_span[1] + offset) * SAMPLE_RATE))]
                np_data = np.array(cur_data)

                mid_word_start = int((offset + left_dur) * SAMPLE_RATE)
                mid_word_end = int((offset + left_dur + mid_dur) * SAMPLE_RATE)

                np_data[mid_word_start: mid_word_end] *= mid_word_volume_factor

                cur_path = f'data/global/long_trios/long_trio{global_audio_id}.wav'
                soundfile.write(file=cur_path,
                                data=np_data,
                                samplerate=int(SAMPLE_RATE))

                classes_to_log.append(AudioData(my_path=cur_path,
                                                orig_path=orig_path,
                                                splices=[int(offset * SAMPLE_RATE),
                                                         mid_word_start,
                                                         mid_word_end,
                                                         int((offset + left_dur + mid_dur + right_dur) * SAMPLE_RATE)]))
                global_audio_id += 1


def save_cur_pair(first, second, offset, loaded_data):
    global global_audio_id
    dur1 = first[1] - first[0]
    dur2 = second[1] - second[0]

    new_data = np.delete(loaded_data, slice(first[0], first[1]))
    new_data = np.delete(new_data, slice(second[0] - dur1, second[1] - dur1))
    new_data = np.insert(new_data, first[0], loaded_data[second[0]: second[1]])
    new_data = np.insert(new_data, second[0] + (dur2 - dur1), loaded_data[first[0]: first[1]])

    new_first = (first[0], first[0] + dur2)
    new_second = (second[1] - dur1, second[1])

    orig1_data = loaded_data[
                 max(0, first[0] - SAMPLE_RATE * offset): min(len(loaded_data) - 1,
                                                              first[1] + SAMPLE_RATE * offset)]
    path_orig1 = f'{PATH_TO_ORIGS}/orig{global_audio_id}.wav'
    global_duration_coefficients.append(
        (path_orig1, (dur1 / SAMPLE_RATE + offset * 2) / len(get_words_in_span(first[0] / SAMPLE_RATE - offset,
                                                                               first[1] / SAMPLE_RATE + offset))))
    soundfile.write(file=path_orig1,
                    data=orig1_data,
                    samplerate=int(SAMPLE_RATE))

    span1 = str(SAMPLE_RATE * offset), str(SAMPLE_RATE * offset + dur2)
    path_swapped1 = f'data/global/swapped/swapped{global_audio_id}.wav'
    soundfile.write(file=path_swapped1,
                    data=new_data[
                         int(max(0, new_first[0] - SAMPLE_RATE * offset)): int(min(len(loaded_data) - 1,
                                                                                   new_first[
                                                                                       1] + SAMPLE_RATE * offset))],
                    samplerate=int(SAMPLE_RATE))  # first

    global_audio_id += 1

    orig2_data = loaded_data[
                 max(0, second[0] - SAMPLE_RATE * offset): min(len(loaded_data) - 1,
                                                               second[1] + SAMPLE_RATE * offset)]
    path_orig2 = f'{PATH_TO_ORIGS}/orig{global_audio_id}.wav'
    global_duration_coefficients.append(
        (path_orig2, (dur2 / SAMPLE_RATE + offset * 2) / len(get_words_in_span(second[0] / SAMPLE_RATE - offset,
                                                                               second[1] / SAMPLE_RATE + offset))))

    soundfile.write(file=path_orig2,
                    data=orig2_data,
                    samplerate=int(SAMPLE_RATE))

    span2 = str(SAMPLE_RATE * offset), str(SAMPLE_RATE * offset + dur1)
    path_swapped2 = f'data/global/swapped/swapped{global_audio_id}.wav'
    soundfile.write(file=path_swapped2,
                    data=new_data[
                         max(0, new_second[0] - SAMPLE_RATE * offset): min(len(loaded_data) - 1,
                                                                           new_second[1] + SAMPLE_RATE * offset)],
                    samplerate=int(SAMPLE_RATE))  # second

    global_audio_id += 1
    pass


def produce_words(audio_path: str, offset: int,
                  shuffling=False,
                  long_words=False,
                  global_dict=False):
    global global_audio_id, global_duration_coefficients

    print('making dict...')
    loaded_data, sr = librosa.load(path=audio_path, sr=SAMPLE_RATE)
    wd = determine_words(audio_path, zipped=shuffling or long_words)

    if shuffling:
        shuffle_swap(loaded_data, audio_path, wd, output_files=5, offset=offset,
                     words_to_add=randint(40, 100))
        return

    if long_words:
        pick_long_trio(loaded_data, audio_path, wd, mid_word_volume_factor=4, offset=offset)
        return

    if global_dict:
        create_global_dict(loaded_data, wd)
        return

    keys = list(filter(lambda f: len(wd[f]) >= 2 and len(f) >= 3, wd.keys()))
    print('swapping..')
    for _ in tqdm(keys):
        selected_key = choice(keys)

        first, second = get_pair_to_swap(wd, selected_key)
        save_cur_pair(first, second, offset, loaded_data)


def save_log_file(log_file_path: str):
    classes_to_log.to_json_file(log_file_path, indent=4)


def merge_audios(dir_path, n, bucket_size, output_path):
    assert bucket_size <= n

    buckets = []
    cur_bucket = []
    paths = []
    print('merging...')
    files = os.listdir(dir_path)
    shuffle(files)
    for i, name in tqdm(enumerate(files)):
        if i >= n:
            break
        loaded_data, sr = librosa.load(path=f'{dir_path}/{name}', sr=SAMPLE_RATE)
        cur_bucket.extend(loaded_data)
        if (i != 0) and (i % bucket_size == 0) or i == n - 1:
            buckets.append(cur_bucket)
            cur_bucket = []
    print('saving buckets...')
    for i, bucket in tqdm(enumerate(buckets)):
        soundfile.write(file=f'{output_path}/bucket{i}.wav',
                        data=bucket,
                        samplerate=SAMPLE_RATE)
        paths.append(f'{output_path}/bucket{i}.wav')
    return paths


def get_words_in_span(start, end):
    global global_cur_determined_words
    return list(filter(lambda f: start <= f[1][0] <= end and start <= f[1][1] <= end, global_cur_determined_words))


def get_longest_audios(top_amount: int):
    global global_cur_determined_words
    return sorted(global_duration_coefficients, key=lambda f: -f[1])[:top_amount]


def clear_dir(p: str):
    print('clearing dir...')
    for d_p in tqdm(os.listdir(p)):
        tqdm(list(map(lambda f: os.remove(f'{p}/{d_p}/{f}'), os.listdir(f'{p}/{d_p}'))))


merged_dir_path = 'data/global/buckets'
clear_dir('data/global')
merged_orig_files = merge_audios(dir_path='data/wavs',
                                 n=1,
                                 bucket_size=1,
                                 output_path=merged_dir_path)
for path in merged_orig_files:
    produce_words(audio_path=path,
                  offset=0,
                  global_dict=True)

aggressive_merging()

save_log_file('data/log.json')