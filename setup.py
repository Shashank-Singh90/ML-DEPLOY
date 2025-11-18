"""Setup configuration for IoT Threat Detection API."""
from setuptools import setup, find_packages

setup(
    name="iot-threat-detection-api",
    version="1.0.0",
    description="IoT network threat detection using machine learning",
    author="Development Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.110.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.1.0",
        "numpy>=1.26.0",
        "prometheus-client>=0.17.0",
        "uvicorn[standard]>=0.27.0",
        "gunicorn>=21.2.0",
        "mlflow>=2.9.2",
        "joblib>=1.3.0",
        "slowapi>=0.1.9",
        "python-multipart>=0.0.6",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "flake8>=6.1.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "iot-threat-api=app.main:run",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)