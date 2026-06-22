INSTRUCTIONS = '''
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
'''

PROMPT_TEMPLATE = '''
QUESTION: {question}

CONTEXT:
{context}
'''.strip()


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE, #course='llm-zoomcamp' I deleted that line as we do not filter through the course name
        model='gemini-2.5-flash'
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions #self.course = course
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        return self.index.search(
            query,
            num_results=num_results
        )

    def build_context(self, search_results):
        lines = []
        for doc in search_results:
            lines.append(f"Filename: {doc['filename']}")
            lines.append(f"Content:\n{doc['content']}")
            lines.append('')
        return '\n'.join(lines).strip()


    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    
    def llm(self, prompt):
        # Generate the response
        response = self.llm_client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                'system_instruction': self.instructions
            }
        )
        
        # Return the ENTIRE response object instead of just response.text
        return response

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        
        # Call llm to get the full response object
        response = self.llm(prompt)
        
        # Extract the text and the usage data
        answer = response.text
        usage = response.usage_metadata
        
        # Return both as a tuple
        return answer, usage