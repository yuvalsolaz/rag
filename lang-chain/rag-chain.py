from langchain.llms import OpenAI

llm = OpenAI(model_name='text-davinci-003')
res = llm('explain langchain components in a few sentences')
print(res)
