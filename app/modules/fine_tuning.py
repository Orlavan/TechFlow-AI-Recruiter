"""
Fine-Tuning Module
Prepares training data and manages fine-tuning process for the Exit Advisor.
"""

import json
import os
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def prepare_exit_advisor_data(
    conversations_path: str = "sms_conversations.json",
    output_path: str = "fine_tuning_data/exit_advisor_training.jsonl"
) -> str:
    """
    Prepares training data for fine-tuning the Exit Advisor model.
    Extracts examples where the action is END or CONTINUE from labeled conversations.

    Args:
        conversations_path: Path to labeled conversations JSON
        output_path: Path to save JSONL training file

    Returns:
        Path to the created training file
    """
    # Load conversations
    with open(conversations_path, 'r', encoding='utf-8') as f:
        conversations = json.load(f)

    training_data = []

    for conv in conversations:
        history = ""

        for turn in conv['turns']:
            speaker = "Recruiter" if turn['speaker'] == 'recruiter' else "Candidate"
            text = turn['text']

            # Look for labeled turns
            if turn.get('label'):
                label = turn['label'].upper()

                # Focus on END vs CONTINUE decisions
                if label in ['END', 'CONTINUE']:
                    training_data.append({
                        "messages": [
                            {
                                "role": "system",
                                "content": """You are an Exit Detection Advisor for a recruitment chatbot.
Determine if the conversation should END or CONTINUE.

END: Candidate not interested, asks to stop, or interview confirmed
CONTINUE: Candidate engaged, asking questions, or discussing scheduling

Output only: END or CONTINUE"""
                            },
                            {
                                "role": "user",
                                "content": f"Conversation:\n{history}\n\nLatest message: {text}\n\nDecision:"
                            },
                            {
                                "role": "assistant",
                                "content": label
                            }
                        ]
                    })

            # Update history
            history += f"{speaker}: {text}\n"

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save as JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Created {len(training_data)} training examples")
    print(f"Saved to: {output_path}")

    return output_path


def prepare_main_agent_data(
    conversations_path: str = "sms_conversations.json",
    output_path: str = "fine_tuning_data/main_agent_training.jsonl"
) -> str:
    """
    Prepares training data for fine-tuning the Main Agent.
    Includes all three actions: CONTINUE, SCHEDULE, END.

    Args:
        conversations_path: Path to labeled conversations JSON
        output_path: Path to save JSONL training file

    Returns:
        Path to the created training file
    """
    with open(conversations_path, 'r', encoding='utf-8') as f:
        conversations = json.load(f)

    training_data = []

    for conv in conversations:
        history = ""

        for turn in conv['turns']:
            speaker = "Recruiter" if turn['speaker'] == 'recruiter' else "Candidate"
            text = turn['text']

            if turn.get('label'):
                label = turn['label'].upper()

                training_data.append({
                    "messages": [
                        {
                            "role": "system",
                            "content": """You are the Main Orchestrator for a recruitment chatbot.
Decide the next action based on the conversation.

Actions:
- CONTINUE: Candidate is engaged, asking questions, or providing info
- SCHEDULE: Candidate wants to schedule, agrees to meet, or proposes a time
- END: Candidate not interested, or interview is confirmed

Output only: CONTINUE, SCHEDULE, or END"""
                        },
                        {
                            "role": "user",
                            "content": f"Conversation:\n{history}\n\nAction:"
                        },
                        {
                            "role": "assistant",
                            "content": label
                        }
                    ]
                })

            history += f"{speaker}: {text}\n"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Created {len(training_data)} training examples")
    print(f"Saved to: {output_path}")

    return output_path


def upload_training_file(file_path: str) -> str:
    """
    Uploads training file to OpenAI.

    Args:
        file_path: Path to JSONL training file

    Returns:
        File ID from OpenAI
    """
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    with open(file_path, 'rb') as f:
        response = client.files.create(
            file=f,
            purpose='fine-tune'
        )

    print(f"File uploaded: {response.id}")
    return response.id


def create_fine_tuning_job(
    training_file_id: str,
    model: str = "gpt-3.5-turbo",
    suffix: str = "exit-advisor"
) -> str:
    """
    Creates a fine-tuning job on OpenAI.

    Args:
        training_file_id: ID of uploaded training file
        model: Base model to fine-tune
        suffix: Suffix for the fine-tuned model name

    Returns:
        Fine-tuning job ID
    """
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model=model,
        suffix=suffix
    )

    print(f"Fine-tuning job created: {response.id}")
    print(f"Status: {response.status}")

    return response.id


def check_fine_tuning_status(job_id: str) -> Dict:
    """
    Checks the status of a fine-tuning job.

    Args:
        job_id: Fine-tuning job ID

    Returns:
        Job status dictionary
    """
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    response = client.fine_tuning.jobs.retrieve(job_id)

    status = {
        'id': response.id,
        'status': response.status,
        'model': response.model,
        'fine_tuned_model': response.fine_tuned_model,
        'created_at': response.created_at,
        'finished_at': response.finished_at
    }

    print(f"Job ID: {status['id']}")
    print(f"Status: {status['status']}")

    if status['fine_tuned_model']:
        print(f"Fine-tuned model: {status['fine_tuned_model']}")

    return status


def list_fine_tuning_jobs(limit: int = 10) -> List[Dict]:
    """
    Lists recent fine-tuning jobs.

    Args:
        limit: Maximum number of jobs to return

    Returns:
        List of job dictionaries
    """
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    response = client.fine_tuning.jobs.list(limit=limit)

    jobs = []
    for job in response.data:
        jobs.append({
            'id': job.id,
            'status': job.status,
            'model': job.model,
            'fine_tuned_model': job.fine_tuned_model,
            'created_at': job.created_at
        })

    return jobs


def run_fine_tuning_pipeline(
    conversations_path: str = "sms_conversations.json",
    model_type: str = "exit_advisor"
):
    """
    Runs the complete fine-tuning pipeline.

    Args:
        conversations_path: Path to labeled conversations
        model_type: 'exit_advisor' or 'main_agent'
    """
    print("=" * 50)
    print(f"Fine-Tuning Pipeline: {model_type}")
    print("=" * 50)

    # Step 1: Prepare data
    print("\n1. Preparing training data...")
    if model_type == "exit_advisor":
        training_file = prepare_exit_advisor_data(conversations_path)
    else:
        training_file = prepare_main_agent_data(conversations_path)

    # Step 2: Upload file
    print("\n2. Uploading training file...")
    file_id = upload_training_file(training_file)

    # Step 3: Create fine-tuning job
    print("\n3. Creating fine-tuning job...")
    job_id = create_fine_tuning_job(
        file_id,
        model="gpt-3.5-turbo",
        suffix=model_type.replace("_", "-")
    )

    print("\n" + "=" * 50)
    print("Fine-tuning job submitted!")
    print(f"Job ID: {job_id}")
    print("\nUse check_fine_tuning_status(job_id) to monitor progress.")
    print("Once complete, update .env with the fine-tuned model ID.")
    print("=" * 50)

    return job_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fine-tuning data preparation')
    parser.add_argument('--prepare', action='store_true', help='Prepare training data')
    parser.add_argument('--upload', type=str, help='Upload a training file')
    parser.add_argument('--start', action='store_true', help='Start fine-tuning pipeline')
    parser.add_argument('--status', type=str, help='Check status of a job')
    parser.add_argument('--list', action='store_true', help='List fine-tuning jobs')

    args = parser.parse_args()

    if args.prepare:
        print("Preparing Exit Advisor training data...")
        prepare_exit_advisor_data()
        print("\nPreparing Main Agent training data...")
        prepare_main_agent_data()

    elif args.upload:
        upload_training_file(args.upload)

    elif args.start:
        run_fine_tuning_pipeline()

    elif args.status:
        check_fine_tuning_status(args.status)

    elif args.list:
        jobs = list_fine_tuning_jobs()
        for job in jobs:
            print(f"{job['id']} - {job['status']} - {job['fine_tuned_model'] or 'N/A'}")

    else:
        print("Use --prepare to create training data")
        print("Use --start to run the fine-tuning pipeline")
        print("Use --status JOB_ID to check job status")
        print("Use --list to see recent jobs")
