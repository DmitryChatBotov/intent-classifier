import re

import numpy as np
import pandas as pd
from Levenshtein import distance as levenshtein_distance


class LevenshteinClassifier:
    def __init__(self, phrases: pd.DataFrame):
        self._punctuation_regexp = re.compile(r"[^\w\s]")
        self._phrases = phrases

    def remove_punctuation_re(self, text):
        """
        Removes punctuation from a given text using regular expressions.

        :param text: A string from which punctuation will be removed.
        :return: The text with punctuation removed.
        """
        return self._punctuation_regexp.sub("", text)

    def classify_question(self, question):
        """
        Classifies an incoming question based on its Levenshtein distance to known intent phrases.

        :param question: The incoming question to classify.
        :return: The ID of the intent that best matches the question or a message indicating no match.
        """
        min_distance = np.inf
        closest_intent = None
        closest_phrase = None
        processed_question = self.remove_punctuation_re(question.lower())

        for _, row in self._phrases.iterrows():
            dist = levenshtein_distance(row["phrase"], processed_question)
            if dist < min_distance:
                min_distance = dist
                closest_intent = row["intent_path"]
                closest_phrase = row["phrase"]
        
        return closest_intent, closest_phrase, min_distance