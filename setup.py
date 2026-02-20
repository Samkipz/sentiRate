from setuptools import setup

# A minimal package solely to install our imghdr shim into the environment.
# This lets the Streamlit CLI import `imghdr` before it ever loads the app
# code; without this the shim sitting in the repo root isn't on sys.path when
# the CLI starts.

setup(
    name="senti_dashboard_shim",
    version="0.1.0",
    py_modules=["imghdr"],
    author="",
    author_email="",
    description="Install an imghdr shim for Python 3.13 compatibility",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
