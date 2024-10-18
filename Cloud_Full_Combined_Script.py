
import boto3
import os
from botocore.exceptions import NoCredentialsError

# AWS S3 Integration for model storage
def upload_model_to_s3(file_name, bucket, object_name=None):
    # Upload a file to an S3 bucket
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name or file_name)
        print(f"File uploaded successfully to S3: {file_name}")
        return True
    except FileNotFoundError:
        print("The file was not found.")
        return False
    except NoCredentialsError:
        print("Credentials not available.")
        return False

# Deploy Cloud to AWS EC2
def deploy_to_ec2(instance_type='t2.micro', key_name='your-key-name', security_group='your-security-group'):
    ec2 = boto3.resource('ec2')

    # Create a new EC2 instance
    instances = ec2.create_instances(
        ImageId='ami-0c55b159cbfafe1f0',  # Example AMI, you can replace with your preferred AMI
        MinCount=1,
        MaxCount=1,
        InstanceType=instance_type,
        KeyName=key_name,
        SecurityGroupIds=[security_group]
    )
    for instance in instances:
        print(f"Created instance with ID: {instance.id}")
    return instances

# Function to set up auto-scaling on AWS
def setup_auto_scaling(auto_scaling_group_name, launch_configuration_name, max_size=5, min_size=1, desired_capacity=2):
    client = boto3.client('autoscaling')

    # Create Auto Scaling group
    response = client.create_auto_scaling_group(
        AutoScalingGroupName=auto_scaling_group_name,
        LaunchConfigurationName=launch_configuration_name,
        MinSize=min_size,
        MaxSize=max_size,
        DesiredCapacity=desired_capacity,
        VPCZoneIdentifier='subnet-your-subnet-id',  # Replace with your subnet ID
        Tags=[
            {
                'Key': 'Name',
                'Value': 'Cloud-AI-Scaling'
            },
        ]
    )
    print("Auto-scaling group created successfully.")
    return response

# Example usage of AWS S3 and EC2 integration
def main():
    # Upload a model file to S3
    model_file = "model.h5"  # Example model file
    bucket_name = "your-bucket-name"
    upload_model_to_s3(model_file, bucket_name)

    # Deploy Cloud to an EC2 instance
    instances = deploy_to_ec2(instance_type='t2.micro', key_name='your-key-name', security_group='your-security-group')

    # Set up auto-scaling for Cloud on AWS
    setup_auto_scaling(auto_scaling_group_name='cloud-auto-scaling-group', launch_configuration_name='cloud-launch-config')

if __name__ == "__main__":
    main()



import boto3
import os
from botocore.exceptions import NoCredentialsError

# AWS S3 Integration for model storage
def upload_model_to_s3(file_name, bucket, object_name=None):
    # Upload a file to an S3 bucket
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name or file_name)
        print(f"File uploaded successfully to S3: {file_name}")
        return True
    except FileNotFoundError:
        print("The file was not found.")
        return False
    except NoCredentialsError:
        print("Credentials not available.")
        return False

# Deploy Cloud to AWS EC2
def deploy_to_ec2(instance_type='t2.micro', key_name='your-key-name', security_group='your-security-group'):
    ec2 = boto3.resource('ec2')

    # Create a new EC2 instance
    instances = ec2.create_instances(
        ImageId='ami-0c55b159cbfafe1f0',  # Example AMI, you can replace with your preferred AMI
        MinCount=1,
        MaxCount=1,
        InstanceType=instance_type,
        KeyName=key_name,
        SecurityGroupIds=[security_group]
    )
    for instance in instances:
        print(f"Created instance with ID: {instance.id}")
    return instances

# Function to set up auto-scaling on AWS
def setup_auto_scaling(auto_scaling_group_name, launch_configuration_name, max_size=5, min_size=1, desired_capacity=2):
    client = boto3.client('autoscaling')

    # Create Auto Scaling group
    response = client.create_auto_scaling_group(
        AutoScalingGroupName=auto_scaling_group_name,
        LaunchConfigurationName=launch_configuration_name,
        MinSize=min_size,
        MaxSize=max_size,
        DesiredCapacity=desired_capacity,
        VPCZoneIdentifier='subnet-your-subnet-id',  # Replace with your subnet ID
        Tags=[
            {
                'Key': 'Name',
                'Value': 'Cloud-AI-Scaling'
            },
        ]
    )
    print("Auto-scaling group created successfully.")
    return response

# Example usage of AWS S3 and EC2 integration
def main():
    # Upload a model file to S3
    model_file = "model.h5"  # Example model file
    bucket_name = "your-bucket-name"
    upload_model_to_s3(model_file, bucket_name)

    # Deploy Cloud to an EC2 instance
    instances = deploy_to_ec2(instance_type='t2.micro', key_name='your-key-name', security_group='your-security-group')

    # Set up auto-scaling for Cloud on AWS
    setup_auto_scaling(auto_scaling_group_name='cloud-auto-scaling-group', launch_configuration_name='cloud-launch-config')

if __name__ == "__main__":
    main()



import psutil
import matplotlib.pyplot as plt
import time

# Function to monitor CPU and memory usage
def monitor_system_resources():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.percent
    return cpu_usage, memory_usage

# Function to monitor model performance
def monitor_model_performance(accuracy):
    if accuracy < 0.85:
        print(f"Alert: Model accuracy dropped to {accuracy:.4f}. Consider retraining.")
    else:
        print(f"Model accuracy is {accuracy:.4f}. All good!")

# Function to set up the alerting system based on CPU and memory thresholds
def alert_system(cpu_usage, memory_usage):
    if cpu_usage > 90:
        print(f"Alert: CPU usage is critically high at {cpu_usage}%!")
    if memory_usage > 90:
        print(f"Alert: Memory usage is critically high at {memory_usage}%!")

# Function to update real-time dashboard
def update_dashboard(cpu_usage, memory_usage, accuracies):
    plt.ion()  # Enable interactive mode
    fig, ax1 = plt.subplots()

    # Plot CPU and memory usage
    ax1.set_xlabel('Time')
    ax1.set_ylabel('CPU Usage (%)', color='tab:blue')
    ax1.plot(range(len(cpu_usage)), cpu_usage, color='tab:blue', marker='o', label='CPU Usage')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()  # Instantiate a second y-axis
    ax2.set_ylabel('Memory Usage (%)', color='tab:red')
    ax2.plot(range(len(memory_usage)), memory_usage, color='tab:red', marker='x', label='Memory Usage')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Plot Model Accuracies
    if accuracies:
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))  # Shift the third axis outward
        ax3.set_ylabel('Model Accuracy', color='tab:green')
        ax3.plot(range(len(accuracies)), accuracies, color='tab:green', marker='^', label='Accuracy')
        ax3.tick_params(axis='y', labelcolor='tab:green')

    fig.tight_layout()  # Ensure subplots don't overlap
    plt.show()
    plt.pause(0.1)  # Pause to allow updates

# Example long-term monitoring and alerting pipeline
def monitoring_pipeline(accuracy_history):
    cpu_usage_history = []
    memory_usage_history = []

    for i in range(10):
        # Monitor system resources
        cpu_usage, memory_usage = monitor_system_resources()

        # Append to history
        cpu_usage_history.append(cpu_usage)
        memory_usage_history.append(memory_usage)

        # Trigger alerts if needed
        alert_system(cpu_usage, memory_usage)

        # Monitor model performance
        current_accuracy = accuracy_history[i] if i < len(accuracy_history) else accuracy_history[-1]
        monitor_model_performance(current_accuracy)

        # Update the dashboard
        update_dashboard(cpu_usage_history, memory_usage_history, accuracy_history[:i+1])

        time.sleep(2)  # Simulate time between monitoring cycles

# Example usage with simulated accuracy history
def main():
    # Simulating accuracy history over time
    accuracy_history = [0.95, 0.92, 0.88, 0.80, 0.85, 0.90, 0.94, 0.89, 0.87, 0.91]

    # Run the monitoring and alerting system
    monitoring_pipeline(accuracy_history)

if __name__ == "__main__":
    main()



import boto3
import os
from botocore.exceptions import NoCredentialsError

# AWS S3 Integration for model storage
def upload_model_to_s3(file_name, bucket, object_name=None):
    # Upload a file to an S3 bucket
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name or file_name)
        print(f"File uploaded successfully to S3: {file_name}")
        return True
    except FileNotFoundError:
        print("The file was not found.")
        return False
    except NoCredentialsError:
        print("Credentials not available.")
        return False

# Deploy Cloud to AWS EC2
def deploy_to_ec2(instance_type='t2.micro', key_name='your-key-name', security_group='your-security-group'):
    ec2 = boto3.resource('ec2')

    # Create a new EC2 instance
    instances = ec2.create_instances(
        ImageId='ami-0c55b159cbfafe1f0',  # Example AMI, you can replace with your preferred AMI
        MinCount=1,
        MaxCount=1,
        InstanceType=instance_type,
        KeyName=key_name,
        SecurityGroupIds=[security_group]
    )
    for instance in instances:
        print(f"Created instance with ID: {instance.id}")
    return instances

# Function to set up auto-scaling on AWS
def setup_auto_scaling(auto_scaling_group_name, launch_configuration_name, max_size=5, min_size=1, desired_capacity=2):
    client = boto3.client('autoscaling')

    # Create Auto Scaling group
    response = client.create_auto_scaling_group(
        AutoScalingGroupName=auto_scaling_group_name,
        LaunchConfigurationName=launch_configuration_name,
        MinSize=min_size,
        MaxSize=max_size,
        DesiredCapacity=desired_capacity,
        VPCZoneIdentifier='subnet-your-subnet-id',  # Replace with your subnet ID
        Tags=[
            {
                'Key': 'Name',
                'Value': 'Cloud-AI-Scaling'
            },
        ]
    )
    print("Auto-scaling group created successfully.")
    return response

# Example usage of AWS S3 and EC2 integration
def main():
    # Upload a model file to S3
    model_file = "model.h5"  # Example model file
    bucket_name = "your-bucket-name"
    upload_model_to_s3(model_file, bucket_name)

    # Deploy Cloud to an EC2 instance
    instances = deploy_to_ec2(instance_type='t2.micro', key_name='your-key-name', security_group='your-security-group')

    # Set up auto-scaling for Cloud on AWS
    setup_auto_scaling(auto_scaling_group_name='cloud-auto-scaling-group', launch_configuration_name='cloud-launch-config')

if __name__ == "__main__":
    main()



import psutil
import matplotlib.pyplot as plt
import time

# Function to monitor CPU and memory usage
def monitor_system_resources():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.percent
    return cpu_usage, memory_usage

# Function to monitor model performance
def monitor_model_performance(accuracy):
    if accuracy < 0.85:
        print(f"Alert: Model accuracy dropped to {accuracy:.4f}. Consider retraining.")
    else:
        print(f"Model accuracy is {accuracy:.4f}. All good!")

# Function to set up the alerting system based on CPU and memory thresholds
def alert_system(cpu_usage, memory_usage):
    if cpu_usage > 90:
        print(f"Alert: CPU usage is critically high at {cpu_usage}%!")
    if memory_usage > 90:
        print(f"Alert: Memory usage is critically high at {memory_usage}%!")

# Function to update real-time dashboard
def update_dashboard(cpu_usage, memory_usage, accuracies):
    plt.ion()  # Enable interactive mode
    fig, ax1 = plt.subplots()

    # Plot CPU and memory usage
    ax1.set_xlabel('Time')
    ax1.set_ylabel('CPU Usage (%)', color='tab:blue')
    ax1.plot(range(len(cpu_usage)), cpu_usage, color='tab:blue', marker='o', label='CPU Usage')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()  # Instantiate a second y-axis
    ax2.set_ylabel('Memory Usage (%)', color='tab:red')
    ax2.plot(range(len(memory_usage)), memory_usage, color='tab:red', marker='x', label='Memory Usage')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Plot Model Accuracies
    if accuracies:
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))  # Shift the third axis outward
        ax3.set_ylabel('Model Accuracy', color='tab:green')
        ax3.plot(range(len(accuracies)), accuracies, color='tab:green', marker='^', label='Accuracy')
        ax3.tick_params(axis='y', labelcolor='tab:green')

    fig.tight_layout()  # Ensure subplots don't overlap
    plt.show()
    plt.pause(0.1)  # Pause to allow updates

# Example long-term monitoring and alerting pipeline
def monitoring_pipeline(accuracy_history):
    cpu_usage_history = []
    memory_usage_history = []

    for i in range(10):
        # Monitor system resources
        cpu_usage, memory_usage = monitor_system_resources()

        # Append to history
        cpu_usage_history.append(cpu_usage)
        memory_usage_history.append(memory_usage)

        # Trigger alerts if needed
        alert_system(cpu_usage, memory_usage)

        # Monitor model performance
        current_accuracy = accuracy_history[i] if i < len(accuracy_history) else accuracy_history[-1]
        monitor_model_performance(current_accuracy)

        # Update the dashboard
        update_dashboard(cpu_usage_history, memory_usage_history, accuracy_history[:i+1])

        time.sleep(2)  # Simulate time between monitoring cycles

# Example usage with simulated accuracy history
def main():
    # Simulating accuracy history over time
    accuracy_history = [0.95, 0.92, 0.88, 0.80, 0.85, 0.90, 0.94, 0.89, 0.87, 0.91]

    # Run the monitoring and alerting system
    monitoring_pipeline(accuracy_history)

if __name__ == "__main__":
    main()



import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to simulate continuous data ingestion
def ingest_continuous_data():
    while True:
        # Simulating incoming batch data
        new_data = pd.DataFrame({
            'feature1': [1.1, 2.2, 3.3],
            'feature2': [4.4, 5.5, 6.6],
            'target': [1, 0, 1]
        })
        yield new_data
        time.sleep(10)  # Simulating time delay between batches

# Function to train or retrain the model
def train_model(data, model=None):
    X = data[['feature1', 'feature2']]
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model is None:
        # Train a new model if none exists
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
    else:
        # Retrain the existing model with new data
        model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy after retraining: {accuracy:.4f}")

    return model, accuracy

# Function to manage long-term deployment with auto-retraining
def long_term_deployment_pipeline(initial_data):
    # Train the initial model
    model, accuracy = train_model(initial_data)

    # Simulate continuous data ingestion
    data_stream = ingest_continuous_data()

    for new_data in data_stream:
        print("New data received. Evaluating model...")

        # Retrain the model if accuracy falls below a threshold
        if accuracy < 0.85:
            print("Accuracy below threshold. Retraining model...")
            combined_data = pd.concat([initial_data, new_data], ignore_index=True)
            model, accuracy = train_model(combined_data, model)
        else:
            print("Model accuracy is sufficient. No retraining needed.")

        # Simulate periodic sleep (e.g., for long-term deployment)
        time.sleep(30)  # Simulate delay between retraining cycles

# Example usage
def main():
    # Simulate initial training data
    initial_data = pd.DataFrame({
        'feature1': [1.2, 2.3, 3.4, 4.5, 5.6],
        'feature2': [6.7, 7.8, 8.9, 9.0, 1.1],
        'target': [1, 0, 1, 0, 1]
    })

    # Deploy Cloud with auto-retraining
    long_term_deployment_pipeline(initial_data)

if __name__ == "__main__":
    main()



import boto3
import os
from botocore.exceptions import NoCredentialsError

# AWS S3 Integration for model storage
def upload_model_to_s3(file_name, bucket, object_name=None):
    # Upload a file to an S3 bucket
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name or file_name)
        print(f"File uploaded successfully to S3: {file_name}")
        return True
    except FileNotFoundError:
        print("The file was not found.")
        return False
    except NoCredentialsError:
        print("Credentials not available.")
        return False

# Deploy Cloud to AWS EC2
def deploy_to_ec2(instance_type='t2.micro', key_name='your-key-name', security_group='your-security-group'):
    ec2 = boto3.resource('ec2')

    # Create a new EC2 instance
    instances = ec2.create_instances(
        ImageId='ami-0c55b159cbfafe1f0',  # Example AMI, you can replace with your preferred AMI
        MinCount=1,
        MaxCount=1,
        InstanceType=instance_type,
        KeyName=key_name,
        SecurityGroupIds=[security_group]
    )
    for instance in instances:
        print(f"Created instance with ID: {instance.id}")
    return instances

# Function to set up auto-scaling on AWS
def setup_auto_scaling(auto_scaling_group_name, launch_configuration_name, max_size=5, min_size=1, desired_capacity=2):
    client = boto3.client('autoscaling')

    # Create Auto Scaling group
    response = client.create_auto_scaling_group(
        AutoScalingGroupName=auto_scaling_group_name,
        LaunchConfigurationName=launch_configuration_name,
        MinSize=min_size,
        MaxSize=max_size,
        DesiredCapacity=desired_capacity,
        VPCZoneIdentifier='subnet-your-subnet-id',  # Replace with your subnet ID
        Tags=[
            {
                'Key': 'Name',
                'Value': 'Cloud-AI-Scaling'
            },
        ]
    )
    print("Auto-scaling group created successfully.")
    return response

# Example usage of AWS S3 and EC2 integration
def main():
    # Upload a model file to S3
    model_file = "model.h5"  # Example model file
    bucket_name = "your-bucket-name"
    upload_model_to_s3(model_file, bucket_name)

    # Deploy Cloud to an EC2 instance
    instances = deploy_to_ec2(instance_type='t2.micro', key_name='your-key-name', security_group='your-security-group')

    # Set up auto-scaling for Cloud on AWS
    setup_auto_scaling(auto_scaling_group_name='cloud-auto-scaling-group', launch_configuration_name='cloud-launch-config')

if __name__ == "__main__":
    main()



import psutil
import matplotlib.pyplot as plt
import time

# Function to monitor CPU and memory usage
def monitor_system_resources():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.percent
    return cpu_usage, memory_usage

# Function to monitor model performance
def monitor_model_performance(accuracy):
    if accuracy < 0.85:
        print(f"Alert: Model accuracy dropped to {accuracy:.4f}. Consider retraining.")
    else:
        print(f"Model accuracy is {accuracy:.4f}. All good!")

# Function to set up the alerting system based on CPU and memory thresholds
def alert_system(cpu_usage, memory_usage):
    if cpu_usage > 90:
        print(f"Alert: CPU usage is critically high at {cpu_usage}%!")
    if memory_usage > 90:
        print(f"Alert: Memory usage is critically high at {memory_usage}%!")

# Function to update real-time dashboard
def update_dashboard(cpu_usage, memory_usage, accuracies):
    plt.ion()  # Enable interactive mode
    fig, ax1 = plt.subplots()

    # Plot CPU and memory usage
    ax1.set_xlabel('Time')
    ax1.set_ylabel('CPU Usage (%)', color='tab:blue')
    ax1.plot(range(len(cpu_usage)), cpu_usage, color='tab:blue', marker='o', label='CPU Usage')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()  # Instantiate a second y-axis
    ax2.set_ylabel('Memory Usage (%)', color='tab:red')
    ax2.plot(range(len(memory_usage)), memory_usage, color='tab:red', marker='x', label='Memory Usage')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Plot Model Accuracies
    if accuracies:
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))  # Shift the third axis outward
        ax3.set_ylabel('Model Accuracy', color='tab:green')
        ax3.plot(range(len(accuracies)), accuracies, color='tab:green', marker='^', label='Accuracy')
        ax3.tick_params(axis='y', labelcolor='tab:green')

    fig.tight_layout()  # Ensure subplots don't overlap
    plt.show()
    plt.pause(0.1)  # Pause to allow updates

# Example long-term monitoring and alerting pipeline
def monitoring_pipeline(accuracy_history):
    cpu_usage_history = []
    memory_usage_history = []

    for i in range(10):
        # Monitor system resources
        cpu_usage, memory_usage = monitor_system_resources()

        # Append to history
        cpu_usage_history.append(cpu_usage)
        memory_usage_history.append(memory_usage)

        # Trigger alerts if needed
        alert_system(cpu_usage, memory_usage)

        # Monitor model performance
        current_accuracy = accuracy_history[i] if i < len(accuracy_history) else accuracy_history[-1]
        monitor_model_performance(current_accuracy)

        # Update the dashboard
        update_dashboard(cpu_usage_history, memory_usage_history, accuracy_history[:i+1])

        time.sleep(2)  # Simulate time between monitoring cycles

# Example usage with simulated accuracy history
def main():
    # Simulating accuracy history over time
    accuracy_history = [0.95, 0.92, 0.88, 0.80, 0.85, 0.90, 0.94, 0.89, 0.87, 0.91]

    # Run the monitoring and alerting system
    monitoring_pipeline(accuracy_history)

if __name__ == "__main__":
    main()



import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to simulate continuous data ingestion
def ingest_continuous_data():
    while True:
        # Simulating incoming batch data
        new_data = pd.DataFrame({
            'feature1': [1.1, 2.2, 3.3],
            'feature2': [4.4, 5.5, 6.6],
            'target': [1, 0, 1]
        })
        yield new_data
        time.sleep(10)  # Simulating time delay between batches

# Function to train or retrain the model
def train_model(data, model=None):
    X = data[['feature1', 'feature2']]
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model is None:
        # Train a new model if none exists
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
    else:
        # Retrain the existing model with new data
        model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy after retraining: {accuracy:.4f}")

    return model, accuracy

# Function to manage long-term deployment with auto-retraining
def long_term_deployment_pipeline(initial_data):
    # Train the initial model
    model, accuracy = train_model(initial_data)

    # Simulate continuous data ingestion
    data_stream = ingest_continuous_data()

    for new_data in data_stream:
        print("New data received. Evaluating model...")

        # Retrain the model if accuracy falls below a threshold
        if accuracy < 0.85:
            print("Accuracy below threshold. Retraining model...")
            combined_data = pd.concat([initial_data, new_data], ignore_index=True)
            model, accuracy = train_model(combined_data, model)
        else:
            print("Model accuracy is sufficient. No retraining needed.")

        # Simulate periodic sleep (e.g., for long-term deployment)
        time.sleep(30)  # Simulate delay between retraining cycles

# Example usage
def main():
    # Simulate initial training data
    initial_data = pd.DataFrame({
        'feature1': [1.2, 2.3, 3.4, 4.5, 5.6],
        'feature2': [6.7, 7.8, 8.9, 9.0, 1.1],
        'target': [1, 0, 1, 0, 1]
    })

    # Deploy Cloud with auto-retraining
    long_term_deployment_pipeline(initial_data)

if __name__ == "__main__":
    main()



import dask.dataframe as dd
from multiprocessing import Pool
import pandas as pd
import time

# Function to simulate parallel data ingestion using multiprocessing
def ingest_parallel_data():
    # Simulate a large dataset split into chunks
    data_chunks = [
        pd.DataFrame({
            'feature1': [1.2, 2.3, 3.4],
            'feature2': [4.5, 5.6, 6.7]
        }),
        pd.DataFrame({
            'feature1': [7.8, 8.9, 9.0],
            'feature2': [1.1, 2.2, 3.3]
        }),
        pd.DataFrame({
            'feature1': [4.4, 5.5, 6.6],
            'feature2': [7.7, 8.8, 9.9]
        })
    ]
    return data_chunks

# Function to preprocess a single chunk of data
def preprocess_data_chunk(chunk):
    # Simulate preprocessing (e.g., scaling, normalization)
    chunk['feature1'] = chunk['feature1'] * 0.5
    chunk['feature2'] = chunk['feature2'] * 0.5
    return chunk

# Function to process each chunk in parallel
def process_data_in_parallel():
    # Ingest data in chunks
    data_chunks = ingest_parallel_data()

    # Set up a multiprocessing pool
    with Pool() as pool:
        # Process each data chunk in parallel
        processed_chunks = pool.map(preprocess_data_chunk, data_chunks)

    # Combine processed chunks into a single dataframe
    combined_data = pd.concat(processed_chunks, ignore_index=True)
    print("Parallel data processing complete. Combined data:")
    print(combined_data)

# Function to distribute a large dataset using Dask
def process_large_data_distributed():
    # Simulate a large dataset
    data = pd.DataFrame({
        'feature1': [i for i in range(10000)],
        'feature2': [i * 2 for i in range(10000)]
    })

    # Convert to Dask dataframe
    ddf = dd.from_pandas(data, npartitions=10)

    # Perform operations on distributed data
    processed_ddf = ddf.map_partitions(lambda df: df * 0.5)
    
    # Trigger computation (lazy evaluation in Dask)
    result = processed_ddf.compute()
    print("Distributed data processing complete.")
    print(result)

# Example usage for parallel and distributed processing
def main():
    print("Running parallel data processing...")
    process_data_in_parallel()

    print("
Running distributed data processing with Dask...")
    process_large_data_distributed()

if __name__ == "__main__":
    main()


