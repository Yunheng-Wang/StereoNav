#!/usr/bin/env python3
"""Custom RxR VLN Dataset that handles the different JSON format."""

import gzip
import json
import os
from typing import TYPE_CHECKING, List, Optional

from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.datasets.utils import VocabDict
from habitat.tasks.nav.nav import NavigationGoal
from habitat.tasks.vln.vln import InstructionData, VLNEpisode

if TYPE_CHECKING:
    from omegaconf import DictConfig


DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"


@registry.register_dataset(name="RxRVLN-v1")
class RxRVLNDataset(Dataset):
    """VLN Dataset class for RxR dataset format (no instruction_vocab)."""

    episodes: List[VLNEpisode]
    instruction_vocab: VocabDict

    @staticmethod
    def check_config_paths_exist(config: "DictConfig") -> bool:
        return os.path.exists(
            config.data_path.format(split=config.split)
        ) and os.path.exists(config.scenes_dir)

    def __init__(self, config: Optional["DictConfig"] = None) -> None:
        self.episodes = []
        # Initialize with empty vocab for RxR
        self.instruction_vocab = VocabDict(word_list=[])

        if config is None:
            return

        dataset_filename = config.data_path.format(split=config.split)
        with gzip.open(dataset_filename, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.scenes_dir)

        self.episodes = list(
            filter(self.build_content_scenes_filter(config), self.episodes)
        )

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        # RxR doesn't have instruction_vocab, keep empty vocab
        if "instruction_vocab" in deserialized:
            self.instruction_vocab = VocabDict(
                word_list=deserialized["instruction_vocab"]["word_list"]
            )

        for ep_data in deserialized["episodes"]:
            # RxR instruction has extra fields (instruction_id, language, etc.)
            # that InstructionData doesn't accept, so extract only needed fields
            instruction_dict = ep_data.get("instruction", {})
            instruction_text = instruction_dict.get("instruction_text", "")
            # Store language info for later filtering
            language = instruction_dict.get("language", "en")

            # Create InstructionData with only supported fields BEFORE creating episode
            instruction_data = InstructionData(instruction_text=instruction_text)
            # Store language as custom attribute for filtering
            instruction_data.language = language

            # Replace the dict with the InstructionData object
            ep_data["instruction"] = instruction_data

            episode = VLNEpisode(**ep_data)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            for g_index, goal in enumerate(episode.goals):
                episode.goals[g_index] = NavigationGoal(**goal)
            self.episodes.append(episode)
