# GenaiPDFReader
The PDF Reader project shows how to transform data from PDFs into embeddings and store them in a vector database. It accepts a query as user input and uses FAISS (Facebook AI Similarity Search) to search for the relevant information from the database.

With the retrieved information as context, it prompts the Gemini Pro model to generate a response. 

[Application with Gemini, Python and PDF inputs](https://medium.com/@akash.hiremath25/unlocking-the-power-of-intelligence-building-an-application-with-gemini-python-and-faiss-for-eb9a055d2429)

## References
To run the file, without making any changes to the environment path file, use : `python -m streamlit run app.py`

[How to Work With a PDF in Python](https://realpython.com/pdf-python/)

[Working with PDF files in Python](https://www.geeksforgeeks.org/working-with-pdf-files-in-python/)

[LangChain's RecursiveCharacterTextSplitter](https://dev.to/eteimz/understanding-langchains-recursivecharactertextsplitter-2846)

[LangChain Docs : Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)

[AttributeError: module 'langchain_community.faiss'](https://stackoverflow.com/a/77885342/11371844)

[Visualise pickle files in Visual Studio Code](https://stackoverflow.com/questions/61124546/is-there-a-way-to-visualize-pickle-files-in-visual-studio-code)

[How to run Python streamlit applications](https://stackoverflow.com/a/74243463/11371844)

[De-serialization relies loading a pickle file](https://stackoverflow.com/q/78120202/11371844)