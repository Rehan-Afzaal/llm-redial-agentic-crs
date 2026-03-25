import json
from Tools import read_dialogue, read_user_data, read_jsonl, read_json, get_conversation_by_id
from tqdm import tqdm
import pandas as pd

# ‘’‘-----------------For example: Reading Movie-------------------------’‘’
path = 'Your_Path/Movie'
final_data_path = '{}/final_data.jsonl'.format(path)
Conversation_path = '{}/Conversation.txt'.format(path)
user_map_path = '{}/user_map.json'.format(path)
item_map_path = '{}/item_map.json'.format(path)


'''Part 1:If you want to go through the whole data'''

final_data = read_jsonl(final_data_path)
user_map = read_json(user_map_path)
item_map = read_json(item_map_path)
Conversation = read_dialogue(Conversation_path)

if __name__ == '__main__':
    '''You have two choices to read the datasets'''

    '''Choices 1:If you want to go through the whole data'''
    Total_len = len(final_data)
    for i in tqdm(range(Total_len), desc='Processing'):
        Per_data = json.loads(final_data[i])
        user_id, user_information = next(iter(Per_data.items()))
        # read user's history_interaction
        history_interaction = user_information['history_interaction']
        # read user_might_likes
        user_might_likes = user_information['user_might_like']
        # read Conversation_info
        Conversation_info = user_information['Conversation']
        # read Conversation Detail Information
        for j in range(len(Conversation_info)):
            per_conversation_info = Conversation_info[j]['conversation_{}'.format(j+1)]
            user_likes_id = per_conversation_info['user_likes']
            user_dislikes_id = per_conversation_info['user_dislikes']
            rec_item_id = per_conversation_info['rec_item']
            # # turn item id into item name, for example:
            for k in range(len(rec_item_id)):
             	rec_name = item_map[rec_item_id[k]]
            # Conversation_id could locate the dialogue
            conversation_id = per_conversation_info['conversation_id']
            # Dialogue
            dialogue = get_conversation_by_id(Conversation, conversation_id)


'''Choice 2:If you want to search for a user's data'''

if __name__ == '__main__':
    user_id = "A30Q8X8B1S3GGT"  # Example User_id
    user_information = read_user_data(final_data_path, user_id)
    # read user's history_interaction
    history_interaction = user_information['history_interaction']
    # read user_might_likes
    user_might_likes = user_information['user_might_like']
    # read Conversation_info
    Conversation_info = user_information['Conversation']
    # read Conversation Detail Information
    for j in range(len(Conversation_info)):
        per_conversation_info = Conversation_info[j]['conversation_{}'.format(j + 1)]
        user_likes_id = per_conversation_info['user_likes']
        user_dislikes_id = per_conversation_info['user_dislikes']
        rec_item_id = per_conversation_info['rec_item']
        # # turn item id into item name, for example:
            for k in range(len(rec_item_id)):
             	rec_name = item_map[rec_item_id[k]]
        # Conversation_id could locate the dialogue
        conversation_id = per_conversation_info['conversation_id']
        # Dialogue
        dialogue = get_conversation_by_id(Conversation, conversation_id)

