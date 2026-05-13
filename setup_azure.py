# 01_setup_azure.py

from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Workspace, AmlCompute

from dotenv import load_dotenv
import os

load_dotenv()


SUBSCRIPTION_ID = os.getenv('SUBSCRIPTION_ID')
RESOURCE_GROUP = os.getenv('RESOURCE_GROUP')
WORKSPACE_NAME = os.getenv('WORKSPACE_NAME')
LOCATION = os.getenv('LOCATION')
COMPUTE_NAME = os.getenv('COMPUTE_NAME')
VM_SIZE = os.getenv('VM_SIZE')


def get_credential():
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
        return credential
    except Exception:
        print("DefaultAzureCredential failed. Opening browser login...")
        return InteractiveBrowserCredential()


def main():
    credential = get_credential()
    
    # MLClient without workspace first, because workspace may not exist yet
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
    )

    print("Creating/updating Azure ML workspace...")

    workspace = Workspace(
        name=WORKSPACE_NAME,
        location=LOCATION,
        display_name=WORKSPACE_NAME,
        description="Workspace for Loan Default MLflow MLOps project",
    )

    ml_client.workspaces.begin_create(workspace).result()
    print(f"Workspace ready: {WORKSPACE_NAME}")

    # Reconnect to workspace
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )

    print("Creating/updating compute cluster...")

    compute = AmlCompute(
        name=COMPUTE_NAME,
        size=VM_SIZE,
        min_instances=0,
        max_instances=1,
        idle_time_before_scale_down=120,
    )

    ml_client.compute.begin_create_or_update(compute).result()
    print(f"Compute ready: {COMPUTE_NAME}")


if __name__ == "__main__":
    main()
