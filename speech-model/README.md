### Running the Docker Container

To run the Docker container with the necessary configurations, use the following command:

```bash
docker run -v /path/to/local/cache:/app/huggingface_cache -p 8005:8005 <image_name>
```

#### Explanation of the Command:
- `-v /path/to/local/cache:/app/huggingface_cache`: This flag mounts a local directory (`/path/to/local/cache`) to the container's directory (`/app/huggingface_cache`). This is useful for caching Hugging Face models locally to avoid downloading them repeatedly.
- `-p 8005:8005`: This flag maps port `8005` on your local machine to port `8005` in the container, allowing you to access the application running inside the container via `http://localhost:8005`.
- `<image_name>`: Replace this placeholder with the name of the Docker image you want to run.

#### Example:
If your local cache directory is `/home/user/huggingface_cache` and your Docker image is `my-speech-model`, the command would look like this:

```bash
docker run -v /home/user/huggingface_cache:/app/huggingface_cache -p 8005:8005 my-speech-model
```

Make sure to replace `/path/to/local/cache` and `<image_name>` with the appropriate values for your setup.