content = open('app.py').read()

# Find and replace the entire predict_single function body
old = '''def predict_single(file_obj, model_choice):
    if file_obj is None:
        return "No file uploaded.", None

    signal = get_signal_from_mat(file_obj)
    if signal is None:
        return "Could not read signal from file.", None'''

new = '''def predict_single(file_obj, model_choice):
    if file_obj is None:
        return "No file uploaded.", None

    signal = get_signal_from_mat(file_obj)
    if signal is None:
        return "Could not read signal from file.", None
    # Ensure signal is long enough
    if len(signal) < WINDOW:
        signal = np.pad(signal, (0, WINDOW - len(signal)))'''

print("Found block 1:", old in content)
content = content.replace(old, new)

# Fix the window extraction + majority vote
old2 = '''    mid = max(0, len(signal)//2 - WINDOW//2)
    window  = signal[mid:mid+WINDOW]
    if len(window) < WINDOW: window = np.pad(window, (0, WINDOW-len(window)))
    model   = cwru_model if model_choice == "CWRU model" else ft_model
    pred, probs, norm = predict_window(model, window)'''

new2 = '''    model = cwru_model if model_choice == "CWRU model" else ft_model

    # Majority vote over 20 evenly spaced windows
    n_vote = 20
    step   = max(STRIDE, len(signal) // n_vote)
    all_probs = []
    for i in range(n_vote):
        start = i * step
        if start + WINDOW > len(signal):
            break
        win = signal[start:start+WINDOW]
        _, p, _ = predict_window(model, win)
        all_probs.append(p)

    avg_probs = np.mean(all_probs, axis=0)
    pred      = int(avg_probs.argmax())
    probs     = avg_probs

    # Display window = first window
    w0     = signal[:WINDOW]
    mean_w = w0.mean(); std_w = w0.std() + 1e-8
    norm   = (w0 - mean_w) / std_w
    window = w0'''

print("Found block 2:", old2 in content)
content = content.replace(old2, new2)

open('app.py', 'w').write(content)
print("Done.")
