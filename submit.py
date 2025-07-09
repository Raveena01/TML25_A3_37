import requests

with open("fgsm_pgd_curriculum_ema_18.pt", "rb") as f:
    response = requests.post(
        "http://34.122.51.94:9090/robustness",
        files={"file": f},
        headers={"token": "42553895", "model-name": "resnet18"}
    )

print(response.json())  # intermediate score (30% samples)
