"""Script to gather Community statistics. Made for Concrete ML but can be easily modified for
another team."""

import os

import requests

if "KEY_FOR_COMMUNITY_DISCOURSE" not in os.environ:
    raise Exception("please get a key KEY_FOR_COMMUNITY_DISCOURSE in your environment")

key_for_community_discourse = os.environ["KEY_FOR_COMMUNITY_DISCOURSE"]

headers = {
    "api-key": key_for_community_discourse,
    "username": "benoit",
}
r = requests.get(
    "https://community.zama.ai/directory_items.json?period=all&order=likes_received&group=ML-team",
    headers=headers,
)
dic = r.json()

if "directory_items" not in dic:
    raise Exception(f"issue, got this dic {dic}")

for statistics, official_name in [
    ("topic_count", "Topics created"),
    ("post_count", "Posts created"),
    ("likes_received", "Likes received"),
    ("solutions", "Solutions"),
]:
    individual = [user[statistics] for user in dic["directory_items"]]
    print(f"    {official_name}: {sum(individual)}")

# For dump in Excel
print("\nDump in Excel (community): \n")

for statistics, _ in [
    ("topic_count", "Topics created"),
    ("post_count", "Posts created"),
    ("likes_received", "Likes received"),
    ("solutions", "Solutions"),
]:
    individual = [user[statistics] for user in dic["directory_items"]]
    print(f"{sum(individual)} \t ", end="")

print()

for statistics, _ in [
    ("topic_count", "Topics created"),
    ("post_count", "Posts created"),
    ("likes_received", "Likes received"),
    ("solutions", "Solutions"),
]:
    individual = [user[statistics] for user in dic["directory_items"]]
    print("0 \t ", end="")

print()
