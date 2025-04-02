# The MIT License (MIT)
# Copyright © 2025 Cazure

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import random
import bittensor as bt
import random
import torch
import base64

from voiceguard.protocol import VoiceGuardSynapse
from voiceguard.validator.stt_reward import get_stt_rewards
from voiceguard.validator.clone_reward import get_clone_rewards
from voiceguard.validator.detection_reward import get_detection_rewards
from voiceguard.utils.uids import get_random_uids
from voiceguard.utils.helper import download_youtube_segment, transcribe_with_whisper, get_video_duration, fetch_random_sentences, get_random_audio_clip, get_random_asv_audio
from voiceguard.utils.misc import select_random_url

async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    # miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
    miner_uids = [1]

    try:
        # Filter out miners with non-functioning axons
        functioning_axons = []
        functioning_uids = []
        for uid in miner_uids:
            axon = self.metagraph.axons[uid] 
            if axon is not None and axon.is_serving:
                functioning_axons.append(axon)
                functioning_uids.append(uid)
        
        if not functioning_uids:
            bt.logging.warning("No functioning axons found among selected miners")
            return

        # equally randomly send requests for tts, cloning, deepfake
        if random.random() < 0: # tts requests 0.333
            random_url = select_random_url()
            duration = get_video_duration(random_url)
            validator_segment = generate_validator_segment(duration)
            output_filepath = download_youtube_segment(random_url, validator_segment)

            if not os.path.exists(output_filepath):
                print("Output file does not exist. Returning empty transcription.")
                transcription = ""
            else:
                transcription = transcribe_with_whisper(output_filepath)

            print("------validator transcript--------")
            print(transcription)
            print("----------------------------------")
            responses = self.dendrite.query(
                axons=functioning_axons,
                synapse = VoiceGuardSynapse(synapse_type="stt", stt_link=random_url, stt_segment=validator_segment),
                deserialize=False,
                timeout=50
            )
            
            rewards = get_stt_rewards(self, query=transcription, responses=responses, time_limit=50)
            print("------after rewards--------")
            self.update_scores(rewards, functioning_uids, score_type="stt")
            
        elif random.random() < 1: # clone request 0.667
            print("--------------------------")
            # read and get clone_audio segment and text to clone from DB
            clone_text = fetch_random_sentences()
            print(f"common_text------: {clone_text}")
            clone_clip_path = get_random_audio_clip()  # Path to the .mp3 file
            print(f"clone_audio_path------: {clone_clip_path}")
            with open(clone_clip_path, "rb") as audio_file:
                clone_clip = audio_file.read() 

            clone_clip_base64 = base64.b64encode(clone_clip).decode("utf-8")
            
            responses = await self.dendrite(
                axons=functioning_axons,
                synapse = VoiceGuardSynapse(synapse_type="clone", clone_clip=clone_clip_base64, clone_text=clone_text),
                deserialize=False,
                timeout=50
            )
            
            rewards = get_clone_rewards(self, clip_audio_path=clone_clip_path, clone_text=clone_text, responses=responses)
            self.update_scores(rewards, functioning_uids, score_type="clone")
            
        elif random.random() < 1: # 1
            # get real audio or fake audio from DB
            random_audio_path, random_audio_type = get_random_asv_audio()
            
            with open(random_audio_path, "rb") as audio_file:
                random_audio = audio_file.read()
            
            responses = self.dendrite.query(
                axons=functioning_axons,
                synapse = VoiceGuardSynapse(synapse_type="detection", detection_audio=random_audio),
                deserialize=False,
                timeout=50
            )
            
            rewards = get_detection_rewards(self, audio_type = random_audio_type, responses=responses, time_limit=50)
            self.update_scores(rewards, functioning_uids, score_type="detection")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        rewards = torch.zeros(len(miner_uids))
        
    bt.logging.info(f"Scored responses: {rewards}")

def generate_validator_segment(duration):
    if duration <= 100:
        return [0, duration]
    else:
        start = random.randint(0, duration - 100)
        return [start, start + 100]