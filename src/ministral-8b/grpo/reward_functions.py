import json
import re
from typing import Any, Dict, List, Union
import zss
from editdistance import eval as edit_distance
from zss import Node
import random

# --- Special Tokens and Regex ---
reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<solution>"
solution_end = "</solution>"

solution_end_regex = r"</solution>[\s]{0,}" + "(?:" + re.escape("<|endoftext|>") + ")?"
match_format_for_solution_extraction = re.compile(
    rf"{re.escape(reasoning_end)}.*?{re.escape(solution_start)}(.+?){solution_end_regex}",
    flags=re.MULTILINE | re.DOTALL
)

# --- Evaluator Class (from Donut) ---
class JSONParseEvaluator:
    """
    Calculate n-TED(Normalized Tree Edit Distance) based accuracy.
    Code adapted from the Donut project by Clova AI Research.
    https://github.com/clovaai/donut/blob/1.0.7/donut/util.py
    """
    @staticmethod
    def update_cost(label1: str, label2: str):
        label1_leaf, label2_leaf = "<leaf>" in label1, "<leaf>" in label2
        if label1_leaf and label2_leaf:
            return edit_distance(label1.replace("<leaf>", ""), label2.replace("<leaf>", ""))
        if label1_leaf != label2_leaf:
            return 1 + len(label1.replace("<leaf>", "") if label1_leaf else label2.replace("<leaf>", ""))
        return int(label1 != label2)

    @staticmethod
    def insert_and_remove_cost(node: Node):
        label = node.label
        return 1 + len(label.replace("<leaf>", "")) if "<leaf>" in label else 1

    def normalize_dict(self, data: Union[Dict, List, Any]):
        if not data: return None
        if isinstance(data, dict):
            new_data = {}
            for key, value in sorted(data.items()):
                value = self.normalize_dict(value)
                if value: new_data[key] = value if not isinstance(value, list) else [value] if len(value) == 1 else value
            return new_data if new_data else None
        if isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                new_data = [item for item in [self.normalize_dict(item) for item in data] if item]
                return sorted(new_data, key=lambda x: json.dumps(x)) if new_data else None
            else:
                return sorted([str(item) for item in data if item is not None and str(item)]) or None
        return str(data)

    def construct_tree_from_dict(self, data: Union[Dict, List], node_name: str = "<root>"):
        node = Node(node_name)
        if isinstance(data, dict):
            for key, value in data.items():
                node.addkid(self.construct_tree_from_dict(value, key))
        elif isinstance(data, list):
            for item in data:
                node.addkid(self.construct_tree_from_dict(item, "<item>"))
        else:
            node.addkid(Node(f"<leaf>{data}"))
        return node

    def cal_acc(self, pred: dict, answer: dict) -> float:
        pred_tree = self.construct_tree_from_dict(self.normalize_dict(pred))
        answer_tree = self.construct_tree_from_dict(self.normalize_dict(answer))

        tree_size = zss.distance(
            self.construct_tree_from_dict(self.normalize_dict({})),
            answer_tree,
            get_children=zss.Node.get_children,
            insert_cost=self.insert_and_remove_cost,
            remove_cost=self.insert_and_remove_cost,
            update_cost=self.update_cost,
        )
        if tree_size == 0:
            return 1.0 if pred_tree.label == answer_tree.label else 0.0

        distance = zss.distance(
            pred_tree,
            answer_tree,
            get_children=zss.Node.get_children,
            insert_cost=self.insert_and_remove_cost,
            remove_cost=self.insert_and_remove_cost,
            update_cost=self.update_cost,
        )
        return max(0, 1 - (distance / tree_size))

# --- Standalone n-TED Calculation Function ---
def calculate_nted(predicted_json: dict, ground_truth_json: dict) -> float:
    """A standalone function to calculate the n-TED score between two JSON objects."""
    evaluator = JSONParseEvaluator()
    return evaluator.cal_acc(predicted_json, ground_truth_json)


# --- Reward Functions for GRPO Training ---
def generate_random_reward(completions, **kwargs):
    """Generates a random reward, useful for establishing a baseline."""
    scores = [1.0 if random.random() < 0.5 else 0.0 for _ in completions]
    return scores

def match_format_approximately(completions, **kwargs):
    """Provides a reward based on whether the generation contains the correct tags."""
    scores = []
    for completion_messages in completions:
        score = 0
        response_content = completion_messages[0]["content"]
        if reasoning_start not in response_content:
            score -= 1.0
        score += 1.0 if response_content.count(reasoning_end)  == 1 else -1.0
        score += 1.0 if response_content.count(solution_start) == 1 else -1.0
        score += 1.0 if response_content.count(solution_end)   == 1 else -1.0
        scores.append(score)
    return scores

def check_json_syntax(completions, **kwargs):
    """Rewards valid JSON syntax within the <solution> tags."""
    scores = []
    for completion_messages in completions:
        response_content = completion_messages[0]["content"]
        match = match_format_for_solution_extraction.search(response_content)
        if match:
            json_part = match.group(1).strip()
            if json_part.startswith("```json"): json_part = json_part[len("```json"):].strip()
            if json_part.endswith("```"): json_part = json_part[:-len("```")].strip()
            print("Extracted json:", json_part)
            try:
                json.loads(json_part)
                scores.append(1.0)
            except json.JSONDecodeError:
                scores.append(-1.0)
        else:
            scores.append(-2.0)
    return scores

def calculate_nted_reward(completions, answer, **kwargs):
    """Rewards based on the n-TED score between the predicted and true JSON."""
    scores = []
    for completion_messages, true_answer_json_str in zip(completions, answer):
        response_content = completion_messages[0]["content"]
        match = match_format_for_solution_extraction.search(response_content)
        print("Found match: ",match)
        if not match:
            scores.append(-2.0)
            continue

        predicted_json_str = match.group(1).strip()
        if predicted_json_str.startswith("```json"): predicted_json_str = predicted_json_str[len("```json"):].strip()
        if predicted_json_str.endswith("```"): predicted_json_str = predicted_json_str[:-len("```")].strip()

        try:
            predicted_json = json.loads(predicted_json_str)
            print("Predicted json: ",predicted_json)
            true_answer_json = json.loads(true_answer_json_str)
            nted_accuracy = calculate_nted(predicted_json, true_answer_json)
            scores.append(nted_accuracy)
        except json.JSONDecodeError:
            scores.append(-1.0)
        except Exception as e:
            print(f"Error during nTED reward calculation: {e}")
            scores.append(-1.5)
    return scores

def check_exact_match(completions, answer, **kwargs):
    """Provides a binary reward for an exact match of the normalized JSON."""
    evaluator = JSONParseEvaluator()
    scores = []
    for completion_messages, true_answer_json_str in zip(completions, answer):
        response_content = completion_messages[0]["content"]
        match = match_format_for_solution_extraction.search(response_content)
        if not match:
            scores.append(0.0)
            continue

        predicted_json_str = match.group(1).strip()
        if predicted_json_str.startswith("```json"): predicted_json_str = predicted_json_str[len("```json"):].strip()
        if predicted_json_str.endswith("```"): predicted_json_str = predicted_json_str[:-len("```")].strip()

        try:
            predicted_json = json.loads(predicted_json_str)
            true_answer_json = json.loads(true_answer_json_str)
            normalized_pred = evaluator.normalize_dict(predicted_json)
            normalized_true = evaluator.normalize_dict(true_answer_json)
            scores.append(1.0 if normalized_pred == normalized_true else 0.0)
        except (json.JSONDecodeError, Exception):
            scores.append(0.0)
    return scores
