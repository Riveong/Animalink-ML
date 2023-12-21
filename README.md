<p align="center">
    <img src="https://github.com/AnimaLink/Machine-Learning-app/assets/91884661/71e483f7-e112-4151-9f30-97c96faac61c"  width="200" height="200">
</p>

# ML Model Deployment

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#endpoint-api">Endpoint API</a></li>
        <li><a href="#built-with">Built With</a></li>
        <li><a href="#api-docs">API Docs</a></li>
      </ul>
    </li>
    <li>
      <a href="#try-this-project">Try This Project</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
  </ol>
</details>

## About The Project
Our Machine Learning (ML) model service is an integral part of our backend system. It powers the Animal Image Classification feature in our mobile application.

### Endpoint API
We have one endpoint that can be used to classify animal images

<table width="100%">
    <tr>
        <th>Method</th>
        <th>Routes</th>
        <th>Type</th>
        <th>Description</th>
    </tr>
    <tr>
        <td colspan="4"><b>Predict</b></td>
    </tr>
     <tr>
        <td>POST</td>
        <td>/predict</td>
        <td>multipart/form-data</td>
        <td>Predict uploaded animals images</td>
    </tr>
</table>

### Built With

- Uvicorn (fastapi)
- Tensorflow
- Pillow
- Python-multipart
- Numpy

### Api Docs 
We employ Swagger UI for effective API documentation, which can be accessed at <a href="#">base_url/docs</a>.

## Try This Project
Dive into this project and discover its functionalities.

### Prerequisites
Before you start, ensure that you have the following software installed on your system:
- Python 3.10.x
- Pip 22.x.x
- Git LFS 

### Installation
1. Clone the repo
   
   ```sh
    git clone https://github.com/AnimaLink/ml-deploy.git
    git lfs pull
   ```
2. Move to `./ml-deploy` directory

   ```sh
    cd ./ml-deploy
   ```
   
3. Install requirements using PIP

   ```sh
    pip install -r requirements.txt
   ```

5. Run ml service application

   ```sh
    python app.py 
   ```
