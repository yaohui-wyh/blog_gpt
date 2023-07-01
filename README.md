# BlogGPT

An AI assistant for automated metadata generation in markdown blogs (e.g. Jekyll/Hexo), also providing content understanding, summarization, etc.

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
python blog_gpt.py -f <path-to-markdown>
```