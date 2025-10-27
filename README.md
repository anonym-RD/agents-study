# Investigating Autonomous Agent Contributions in the Wild: Activity Patterns and Code Change over Time

The rise of large language models for code has reshaped software development. Autonomous coding agents, able to create branches, open pull requests, and perform code reviews, now actively contribute to real-world projects. Their growing role offers a unique and timely opportunity to investigate AI-driven contributions and their effects on code quality, team dynamics, and software maintainability. In this work, we construct a novel dataset of approximately 110,000 open-source pull requests, including associated commits, comments, reviews, issues, and file changes, collectively representing millions of lines of source code. We compare five popular coding agents, including OpenAI Codex, Claude Code, GitHub Copilot, Google Jules, and Devin, examining how their usage differs in various development aspects such as merge frequency, edited file types, and developer interaction signals, including comments and reviews. Furthermore, we emphasize that code authoring and review are only a small part of the larger software engineering process, as the resulting code must also be maintained and updated over time. Hence, we offer several longitudinal estimates of survival and churn rates for agent-generated versus human-authored code. Ultimately, our findings indicate an increasing agent activity in open-source projects, although their contributions are associated with more churn over time compared to human-authored code. 

## Dataset 

We publish the dataset along with the code change results on [HuggingFace](https://huggingface.co/datasets/askancv/agents_activity). Instructions for loading each subset for reproducing the analysis are also provided there.

## Code 

The code for dataset scraping and analysis is available in the *data_scraper*, *activity_analysis*, and *code_change_analysis* directories. 
