from evaluate import load

bleu = load("bleu")
rouge = load("rouge")
# bertscore = load("bertscore")


def compute_bleu(predictions, references):
    return bleu.compute(predictions=predictions, references=references)


def compute_rouge(predictions, references):
    return rouge.compute(predictions=predictions, references=references)


# def compute_bertscore(predictions, references):
#     return bertscore.compute(
#         predictions=predictions,
#         references=references,
#         lang="en",
#         device="cpu",
#         model_type="distilbert-base-uncased",
#     )
