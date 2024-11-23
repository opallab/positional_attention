import json
import random
import os

all_text = "".join([f"Cat{chr(i+1)}" for i in range(1, 41)])
all_text += "Find sum of categories, and"
all_text += "Find min of categories, and"
all_text += "Find max of categories, and"
chars = sorted(list(set(all_text)))

def encode(l: list[any]) -> list[int]:
		"""Encode a list of strings and numbers into a tokenized list of integers."""
		encoded = []
		stoi = {ch: (-(i+1)) for i, ch in enumerate(chars)}
		for s in l:
				if type(s) == str:
						encoded += [stoi[c] for c in s]
				else:
						encoded.append(s)
		return encoded

def generate_sample_prompt(categories: list[str], low: float, high: float, query_type: str, num_query_cats = None) -> dict:
		"""Generate a sample for the expenses dataset.

		Args:
				categories (float): list of category names
				low (float): lower bound for the prices
				high (float): upper bound for the prices
				query_type (str): type of query to generate
				num_cats (int): number of categories to use in the query (only applicable for sum query type)
		"""
		if query_type == "multitask":
				query_type = random.choice(["min", "max"])
		
		ranges = [random.uniform(low, high) for _ in range(2)]
		low, high = min(ranges), max(ranges)
		expenses = {cat: random.uniform(low, high) for cat in categories}

		breakdown = []
		for cat, amount in expenses.items():
				breakdown.append(cat)
				breakdown.append(amount)

		query, query_answer = None, None
		if query_type == "sum":
				if num_query_cats is None:
						raise ValueError("num_query_cats must be specified for sum query type")
				query_categories = random.sample(categories, num_query_cats)
				query = f"Find sum of categories {', '.join(query_categories[:-1])} and {query_categories[-1]}"
				query_answer = sum(expenses[cat] for cat in query_categories)
		elif query_type == "min":
				if num_query_cats is None:
						raise ValueError("num_query_cats must be specified for min query type")
				query_categories = random.sample(categories, num_query_cats)
				query = f"Find min of categories {', '.join(query_categories[:-1])} and {query_categories[-1]}"
				query_answer = min([expenses[cat] for cat in query_categories])
		elif query_type == "max":
				if num_query_cats is None:
						raise ValueError("num_query_cats must be specified for sum query type")
				query_categories = random.sample(categories, num_query_cats)
				query = f"Find max of categories {', '.join(query_categories[:-1])} and {query_categories[-1]}"
				query_answer = max([expenses[cat] for cat in query_categories])
		elif query_type == "sort":
				query = f"Sort the expenses in ascending order."
				query_answer = " ".join(str(val) for val in sorted(expenses.values()))

		prompt = breakdown + [query]
		return {"prompt": prompt, "answer": query_answer}


def generate_tokenized_sample(num_cats: int, query_type: str, low: float, high: float, num_query_cats = None, train: bool = True) -> dict:
		"""Generate a tokenized sample for the expenses dataset."""
		if train:
			cat_nums = random.sample(list(range(1, 21)), num_cats)
		else:
			cat_nums = random.sample(list(range(21, 41)), num_cats)
		cat_nums = sorted(cat_nums)

		categories = []
		for i in cat_nums:
				categories.append(f"Cat{chr(i+1)}")
		
		sample = generate_sample_prompt(categories, low, high, query_type, num_query_cats=num_query_cats)
		return encode(sample["prompt"]), sample["answer"]

