import pickle

with open("data.pkl","rb") as file:
    data = pickle.load(file)

codes = data["codes"]

def label_encoder(symptoms):
    encoded_values = []
    for symptom in symptoms:
        encoded_values.append(codes[symptom])
    n = len(encoded_values)
    if n < 17:
        encoded_values += [0]*(17-n)
    elif n > 17:
        encoded_values = encoded_values[:17]
    else:
        pass
    return encoded_values