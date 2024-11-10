# Steps to build Multi Agent RAG system
## Step 1: Install the dependencies
## Step 2: import the packages
Step 3: assign the OpenAI secret key from the colab secret key
Step 4: Load llm using ChatOpenAI
Step 5: main module :Create vector stores using chroma db.
Step 5a:Load your document files. There are 2 pdf files finance_data and public_data. Objective is to create 2 separate vector stores one for finance_data and other for public_data to improve latency.
Step 5b: Split the loaded documents into chunks. Experiment with different chunking method and parameters and check the final response. Recursive Character Text Splitter is one of the chunking methods. 
Step 5c: Create embeddings and store in Vector Database. Use 'text-embedding-3-small' model.
Step 5d: The vector databases need to stored for future use. These should not be created every time the user does any transaction. Create as a separate module.
Step 6 : App module : Create multiple agents. One is a planning agent who will do the following tasks.
Steps 6a : Call the classification agent to classify the user query into two classes - finance  or public. 
Step 6b : Call the access_check agent to check the access of the user and get the user access authority
Step 6c : Call the data_retriever agent to do the retrieval from the vector stores and do the augmentation and creates the prompt. Experiment with and without MMR and check the final response
Step 6d : Call the Q&A agent to generate the response from the AI model , format it and creates the final response for the user.  
Step 8 : Frame at least five logical questions relevant to the knowledge base and demonstrate relevant answers from the RAG system. Experiment with different combinations of different hyperparameters in all above steps to get good results.
Step 9: Create a UI where user can write the query and get the response from the RAG system. Step 9a : Implement User Access Control is your RAG system for enhanced data protection. The users are assigned roles (e.g., admin, researcher, end-user), and access rights are granted based on these roles. For example : admins have access to finance_data file and public_data file. Researcher and enduser have access to public_data file only.
Step 9b : Users must authenticate themselves, typically through a username and password. New users have to sign up giving a username , password and user type (any one of admin, user). Please save these info in a CSV file and retrieve during authentication.
Step 9c : Once authenticated, the system checks the user’s permissions to determine what actions they can perform, such as accessing certain files or using specific features.
Retrieve information only from the files that the user has access to. Do not use the information where the access is restricted. Please use all packages which are compatible with python 3.12 version. Check the compatibility of all packages strictly. Please create comments before every class or function. The code will be executed in github codespace.


# multi_agent_rag
