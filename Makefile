.PHONY: build debug course course-push

build:
	uv run langgraph build -t langgraph-python-agent:$(TAG)

debug:
	langgraph dev --allow-blocking --debug-port 1234

course:
	uv run python scripts/course.py

course-push:
	@[ -n "$(SHA)" ] || (echo "请提供提交 hash: make course-push SHA=<hash>" && exit 1)
	uv run python scripts/course-push.py $(SHA)
