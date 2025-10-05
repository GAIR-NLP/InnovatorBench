# 🚀 Introduction

This is the preparation for the ResearchGym. It contains a docker image and a web server.

The docker image should be used for the Command Actions. Every computer/container should use this image for multi-computer control and asynchronous execution.

This docker image comtains several important components:
- http_terminal_server.py: A server that can be used to control the terminal of the computer/container.
- Automatically activate the conda environment when the container is started.
- Several important libraries.

The web server should be used for the Web Browser Actions. You can only serve it on one computer! 


# 💻 Build the docker image

```bash
cd <path to this directory>
docker build -t researchgym .
```

Note: The directory should contain both `Dockerfile` and `http_terminal_server.py`.

# 🌐 Build the web server

Choose a machine as the web server(Even outside the cluster is ok). Install the following package:

```txt
aiofiles==24.1.0
blinker==1.9.0
click==8.2.1
exceptiongroup==1.3.0
Flask==3.1.1
greenlet==3.2.3
h11==0.16.0
h2==4.2.0
hpack==4.1.0
Hypercorn==0.17.3
hyperframe==6.1.0
itsdangerous==2.2.0
Jinja2==3.1.6
MarkupSafe==3.0.2
playwright==1.54.0
priority==2.0.0
pyee==13.0.0
Quart==0.20.0
taskgroup==0.2.2
tomli==2.2.1
typing_extensions==4.14.1
Werkzeug==3.1.3
wsproto==1.2.0
```

Run the script:

```bash
python web_server.py
```

Then you can access the web server at `http://<ip>:8124`.


# Update the task environment

Please remember update the parameters about the web server and the docker image in the task environment. 

# Environment in task 20
Task 20 needs vision experts in [VisualSketchpad](https://github.com/Yushi-Hu/VisualSketchpad). We modify some of its file to adjust ResearchGym. You should set the server before running task 20. The installation documents is in [installation.md](./visualsketchpad/vision_experts/installation.md). You must use the `visualsketchpad` in `./visualsketchpad`, but you can save it anywhere.