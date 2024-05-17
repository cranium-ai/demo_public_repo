import boto3
import json

# Set up AWS credentials and region
aws_access_key_id = 'your-access-key-id'
aws_secret_access_key = 'your-secret-access-key'
region_name = 'your-region-name'

# Initialize the AWS Bedrock client
client = boto3.client(
    'bedrock',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)

def generate_response(prompt):
    response = client.invoke_model(
        modelId='text-davinci-003',  # Replace with the appropriate model ID
        contentType='application/json',
        body=json.dumps({
            'prompt': prompt,
            'max_tokens': 150,
            'temperature': 0.7
        })
    )
    response_body = json.loads(response['body'].read())
    return response_body['text']

def chatbot():
    print("Hello! I am an AI chatbot. Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        response = generate_response(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chatbot()
