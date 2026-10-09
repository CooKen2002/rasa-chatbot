import logging
from typing import Any, Dict, List, Text

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from rasa.nlu.featurizers.dense_featurizer.dense_featurizer import DenseFeaturizer
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.constants import TOKENS_NAMES
from rasa.shared.nlu.constants import TEXT
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData

logger = logging.getLogger(__name__)

CLS_TOKEN = "__CLS__"
MAX_LEN = 256


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER, is_trainable=False
)
class PhoBertFeaturizer(DenseFeaturizer, GraphComponent):
    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["torch", "transformers"]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {
            "alias": None,  # Featurizer của Rasa bắt buộc có key này
            "model_weights": "vinai/phobert-base",
        }
    
    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """Không cần kiểm tra gì thêm cho featurizer này."""
        pass
    def __init__(
        self, config: Dict[Text, Any], execution_context: ExecutionContext
    ) -> None:
        config = {**self.get_default_config(), **(config or {})}
        super().__init__(execution_context.node_name, config)
        weights = config["model_weights"]

        logger.info(f"Đang tải PhoBERT từ: {weights}...")
        self.tokenizer = AutoTokenizer.from_pretrained(weights)
        self.model = AutoModel.from_pretrained(weights).eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> "PhoBertFeaturizer":
        return cls(config, execution_context)

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> "PhoBertFeaturizer":
        return cls(config, execution_context)

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        self.process(training_data.training_examples)
        return training_data

    def process(self, messages: List[Message]) -> List[Message]:
        for message in messages:
            self._process_message(message)
        return messages

    def _process_message(self, message: Message) -> None:
        if message.get(TEXT) is None:
            return

        tokens = message.get(TOKENS_NAMES[TEXT]) or []
        words = [t.text for t in tokens if t.text != CLS_TOKEN]
        if not words:
            return

        # Mã hoá từng từ thành subword, nhớ lại số subword của mỗi từ
        bos, eos = self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
        ids: List[int] = [bos]
        spans = []  # (start, end) của mỗi từ trong input_ids
        for w in words:
            piece = self.tokenizer.encode(w, add_special_tokens=False) or [
                self.tokenizer.unk_token_id
            ]
            spans.append((len(ids), len(ids) + len(piece)))
            ids.extend(piece)
        ids.append(eos)

        # Cắt nếu quá dài; các từ bị cắt sẽ nhận vector 0
        ids = ids[: MAX_LEN - 1] + [eos] if len(ids) > MAX_LEN else ids

        input_ids = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            hidden = (
                self.model(
                    input_ids=input_ids, attention_mask=torch.ones_like(input_ids)
                )
                .last_hidden_state[0]
                .cpu()
                .numpy()
            )

        dim = hidden.shape[1]
        seq = np.zeros((len(tokens), dim), dtype=np.float32)
        word_idx = 0
        for i, t in enumerate(tokens):
            if t.text == CLS_TOKEN:
                seq[i] = hidden[0]  # dùng vector <s>
                continue
            s, e = spans[word_idx]
            word_idx += 1
            if s < len(ids) - 1:
                seq[i] = hidden[s : min(e, len(ids) - 1)].mean(axis=0)

        sentence = hidden[0].reshape(1, -1)

        message.add_features(Features(seq, "sequence", TEXT, "PhoBertFeaturizer"))
        message.add_features(Features(sentence, "sentence", TEXT, "PhoBertFeaturizer"))
