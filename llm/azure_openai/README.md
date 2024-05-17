# AI Chatbot using Azure OpenAI

This project demonstrates how to create a simple AI chatbot using Azure OpenAI and Flask. The chatbot takes user input, sends it to the OpenAI API, and returns a generated response.

## Features

- Interact with the chatbot via a web interface.
- Leverage the power of Azure OpenAI for generating responses.
- Easy to deploy on Azure App Service.

## Prerequisites

- [Azure account](https://azure.microsoft.com/)
- Python 3.7 or higher
- Flask

## Setup

### Step 1: Create an Azure OpenAI Resource

1. Go to the [Azure portal](https://portal.azure.com/).
2. Create a new Azure OpenAI resource.
3. Navigate to the resource and copy the API key and endpoint URL.

### Step 2: Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/ai-chatbot-azure-openai.git
cd ai-chatbot-azure-openai
```

### Step 3: Install Dependencies

Install the required Python packages:

`pip install -r requirements.txt`

### Step 4: Set Up Environment Variables

Create a .env file in the project root directory and add your Azure OpenAI API key and endpoint URL:

`OPENAI_API_KEY=your-api-key`

`OPENAI_API_BASE=your-endpoint-url`

### Step 5: Run the Chatbot Locally

Run the Flask application:

`python app.py`

The chatbot will be available at http://127.0.0.1:5000/chat.

### Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes.

### License

This project is licensed under the MIT License. See the LICENSE file for details.


