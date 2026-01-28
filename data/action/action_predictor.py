import os
import json
from openai import OpenAI
import random

client = OpenAI(api_key="YOUR_API_KEY")


def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def call_openai_api(system_prompt, user_prompt):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            top_p=0.1
        )
        return completion.choices[0].message.content.strip().split("\n")
    except Exception as e:
        return [f"Error: {str(e)}"]


def process_scene(scene_data, actions_list):
    """
    Process a single scene and predict human actions.

    Args:
        scene_data: Dictionary containing scene information with keys:
            - scene_id: Unique identifier for the scene
            - scene_type: Type of room (e.g., "Bedroom", "LivingRoom")
            - relations: List of spatial relationships between objects
        actions_list: List of possible actions from action annotations

    Returns:
        Dictionary containing predicted actions for the scene
    """
    scene_relations_json = scene_data.get("relations", [])
    space_type = scene_data.get("scene_type", "unknown")
    scene_id = scene_data.get("scene_id", "unknown")

    scene_relations = "\n".join([
        f'- {rel["subject"]} {rel["relation"]} {rel["object"]}'
        for rel in scene_relations_json
    ])

    random.shuffle(actions_list)
    actions = "\n".join([f'- {action}' for action in actions_list])

    system_prompt = f"""
    You are an assistant that analyzes spatial relationships and identifies possible actions within a given space.

    - The space is described by its type: {space_type}.
    - Spatial relationships are provided in the structure: {{"subject", "relation", "object"}}.

    Your task:
    1. Analyze the spatial relationships and understand how objects are connected within the space.
    2. Identify Relevant Actions:
    - Include a diverse range of actions involving object interactions, human movements, and person-to-person interactions.
    - Actions should reflect typical human activities, object functionality, or interactions likely to occur within the given space.
    - Consider different perspectives and scenarios. For example:
        - Practical interactions (e.g., 'open', 'sit', 'place').
        - Human motion or navigation (e.g., 'walk around', 'reach for').
        - Combined interactions (e.g., 'store items in', 'retrieve from').
    3. Prioritize Specificity:
    - Prioritize actions that demonstrate direct or indirect spatial interactions, emphasizing specific object use or human behavior.
    - Avoid overly abstract or generic actions.
    4. Exclude actions that are loosely related to spatial relationships, such as referencing objects without interaction or actions unrelated to spatial organization.
    Actions that do not result in changes to object placement or spatial organization should be excluded.
    5. Provide only the list of actions.
    6. Please focus on plausible, contextually relevant actions only. Limit your final list to 30 or fewer items, ensuring each action aligns closely with the described scene and spatial relationships.

    Result Format:
    - List all relevant actions in the format:
    - [human, action, object] (e.g., "human sitting on Sofa")
    - [human, action] (e.g., "human watching Television")
    - [object, action, object] (if it involves object-object interaction, e.g., "human place book on shelf")
    """

    user_prompt = f"""
    Given the space type {space_type} and the following spatial relationships between objects:
    {scene_relations}

    Identify all possible actions that could occur in this space based on the relationships provided. Use only the actions listed below:
    {actions}

    List the actions in the format:
    - {{human}} {{subject / object}} {{action}} or {{human}} {{action}}
    """

    predicted_actions = call_openai_api(system_prompt, user_prompt)

    output_data = {
        "scene_id": scene_id,
        "scene_type": space_type,
        "relations": scene_relations_json,
        "predicted_actions": predicted_actions
    }

    return output_data


def process_json_files(input_dir, actions_path, output_dir):
    """
    Process all scene JSON files in input directory and generate action predictions.

    Args:
        input_dir: Directory containing input scene JSON files
        actions_path: Path to action annotations JSON file
        output_dir: Directory to save output files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load action annotations
    raw_actions_json = load_json(actions_path)["label"]
    actions_list = [
        annotation["action"]
        for entry in raw_actions_json
        for annotation in entry["annotations"]
    ]

    all_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    total_files = len(all_files)
    print(f"Total files to process: {total_files}")

    for idx, file_name in enumerate(all_files):
        scene_id = file_name.split('.')[0]
        output_file_path = os.path.join(output_dir, f"{scene_id}_output.json")

        # Skip if output already exists
        if os.path.exists(output_file_path):
            print(f"Skipping existing file: {output_file_path}")
            continue

        try:
            file_path = os.path.join(input_dir, file_name)
            scene_data = load_json(file_path)

            output_data = process_scene(scene_data, actions_list.copy())
            save_json(output_file_path, output_data)

            print(f"Processed [{idx+1}/{total_files}]: {file_name}")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")


if __name__ == "__main__":
    # Example usage
    input_dir = "./input"
    actions_path = "./action_anns.json"
    output_dir = "./output"

    process_json_files(input_dir, actions_path, output_dir)
