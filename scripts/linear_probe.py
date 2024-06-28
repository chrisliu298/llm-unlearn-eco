import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model_name = "zephyr-7b-beta"
# model_name = "Yi-34B-Chat"
# model_name = "Mixtral-8x7B-Instruct-v0.1"
setup = "corrupt_rand_noise_first_n_dims=1-strength=1000000.0"
logits_path = f"results/wmdp/{model_name}_{setup}_logits.pt"
labels_path = f"results/wmdp/{model_name}_{setup}_labels.pt"
logits = torch.load(logits_path)
labels = torch.load(labels_path)

bio_logits = logits["wmdp-bio"]
chem_logits = logits["wmdp-chem"]
cyber_logits = logits["wmdp-cyber"]

bio_labels = labels["wmdp-bio"]
chem_labels = labels["wmdp-chem"]
cyber_labels = labels["wmdp-cyber"]

bio_size, chem_size, cyber_size = (
    bio_logits.size(0),
    chem_logits.size(0),
    cyber_logits.size(0),
)
bio_train_size, chem_train_size, cyber_train_size = (
    int(bio_size * 0.5),
    int(chem_size * 0.5),
    int(cyber_size * 0.5),
)

bio_train_logits, bio_test_logits = (
    bio_logits[:bio_train_size],
    bio_logits[bio_train_size:],
)
chem_train_logits, chem_test_logits = (
    chem_logits[:chem_train_size],
    chem_logits[chem_train_size:],
)
cyber_train_logits, cyber_test_logits = (
    cyber_logits[:cyber_train_size],
    cyber_logits[cyber_train_size:],
)

train_logits = torch.cat(
    [bio_train_logits, chem_train_logits, cyber_train_logits], dim=0
).numpy()
test_logits = torch.cat(
    [bio_test_logits, chem_test_logits, cyber_test_logits], dim=0
).numpy()

bio_train_labels, bio_test_labels = (
    bio_labels[:bio_train_size].numpy(),
    bio_labels[bio_train_size:].numpy(),
)
chem_train_labels, chem_test_labels = (
    chem_labels[:chem_train_size].numpy(),
    chem_labels[chem_train_size:].numpy(),
)
cyber_train_labels, cyber_test_labels = (
    cyber_labels[:cyber_train_size].numpy(),
    cyber_labels[cyber_train_size:].numpy(),
)

del bio_logits, chem_logits, cyber_logits, bio_labels, chem_labels, cyber_labels

for subset, train_logits, train_labels, test_logits, test_labels in [
    ("Bio", bio_train_logits, bio_train_labels, bio_test_logits, bio_test_labels),
    ("Chem", chem_train_logits, chem_train_labels, chem_test_logits, chem_test_labels),
    (
        "Cyber",
        cyber_train_logits,
        cyber_train_labels,
        cyber_test_logits,
        cyber_test_labels,
    ),
]:
    model = LogisticRegression()

    model.fit(train_logits, train_labels)

    y_train_pred = model.predict(train_logits)
    y_test_pred = model.predict(test_logits)

    train_accuracy = accuracy_score(train_labels, y_train_pred)
    test_accuracy = accuracy_score(test_labels, y_test_pred)

    # Print the accuracies
    print(f"{subset} train accuracy: {train_accuracy:.4f}")
    print(f"{subset} test accuracy: {test_accuracy:.4f}")
