# BlogGPT

An AI assistant for automated metadata generation in markdown blogs (e.g. Jekyll/Hexo), also providing content understanding, summarization, etc.

> Developer's note: [A Hands-on Journey to a working LangChain LLM Application](https://wyh.life/article/2023/07/02/llm-app-development)

## 💻 Running Locally

```shell
git clone https://github.com/yaohui-wyh/blog-gpt.git
cd blog-gpt
```

Install dependencies with Poetry and activate virtual environment

```shell
poetry install
poetry shell
```

Run blog_gpt.py to generate metadata for a blog post:

```shell
cd blog-gpt
export OPENAI_API_KEY=<openai-key>

# Generate content summarization & keywords
python blog_gpt.py -f <path-to-markdown-file>

# Q&A about content
python blog_gpt.py -f <path-to-markdown-file> -q <question>
```

### Usage Example

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/from-referrer/)

Summarize and answer questions about the BlogGPT README.md:

```plain
➜ python blog_gpt.py -f ../README.md | jq
{
  "summary": "BlogGPT is an AI assistant that can generate metadata for markdown blogs, as well as provide content understanding and summarization. To use it, clone the repository, install dependencies, and run blog_gpt.py.",
  "keywords": "AI, metadata, markdown, blog, summarization"
}

➜ python blog_gpt.py -f ../README.md -q "can blog_gpt generate keywords"
Yes, blog_gpt can generate keywords.
```
