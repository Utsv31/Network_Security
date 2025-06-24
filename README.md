

# 🛡️ Network Security Project

## 📖 Overview

This project is a **phishing and malicious URL detection system** built using machine learning and deployed using **AWS Lambda**, **API Gateway**, **S3**, **ECR**, and **Docker**. The model is trained on a set of URL and network security features and served as a containerized web service.

---

## 🧠 Dataset

The project is trained on a CSV file containing **31 columns** of URL and network security features.

**Key Columns:**

* `having_IP_Address`, `URL_Length`, `Shortining_Service`, `having_At_Symbol`, `double_slash_redirecting`, `Prefix_Suffix`, `having_Sub_Domain`, `SSLfinal_State`, `Domain_registeration_length`, etc.
* `Result`: Target column (`1` for phishing, `-1` for legitimate).

---

## ⚡️ Deployment Architecture

This project uses:

* 🐳 **Docker** — containerize the application.
* ☁️ **AWS Lambda & API Gateway** — serverless HTTP interface.
* 📦 **AWS S3** — store output and model files.
* 🐙 **GitHub Actions** — CI/CD pipeline to build, test, and deploy automatically.



Deploy using AWS:

Configure your AWS credentials and trigger the GitHub Actions workflow to:

* 📦 Push the Docker image to ECR.
* 🚀 Deploy container to AWS EC2.

---

## 🛠️ Requirements

* 🐍 **Python 3.10+**
* 🐳 **Docker**
* 🔧 **AWS CLI** configured with credentials
* 📜 `requirements.txt` dependencies installed

---

