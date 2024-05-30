[Colab Link](https://colab.research.google.com/drive/152ADFBIZr6j0fMgLpPGvfBC0XEKoY76f?usp=sharing)

[Colab Christophe](https://colab.research.google.com/drive/1QDwWmu8kRxZFI6nRI6nbOu31eYmYEywE?usp=sharing)

## Getting started

To get started, follow these steps:

1. Install Poetry (only for the first execution):
    ```shell
    pip install poetry
    ```

2. Install project dependencies (only for the first execution):
    ```shell
    poetry install
    ```

3. Activate the Virtual Environment:
    ```shell
    poetry shell
    ```


4. Add Dependencies (Optional): If you need to add new dependencies to your project, you can use the following command:
    ```shell
    poetry add <package-name>
    ```

4. Update Dependencies (Optional): To update the dependencies to the latest versions according to the constraints in your pyproject.toml file, use:
    ```shell
    poetry update
    ```