
# ilab-template-python-data-science/requirements directory

Template for python projects tailored to scientific applications (e.g., machine learning) - requirements directory

## Objectives

- Directory for application requirements and dependencies
- Directory to store Dockerfile's to build the application container

## Files Summary

These Dockerfiles get deployed via GitHub Actions. The production container
gets deployed when a release is produced. The development container gets built
when a change is done to the develop branch. You can change this behavior by
making changes to the GitHub Actions file 'dockerhub.yml'.

- Dockerfile: production container
- Dockerfile.dev: devepment container
- requirements.txt: packagages needed to deploy this problem to a notebook environment
