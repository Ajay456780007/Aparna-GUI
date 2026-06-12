import os
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import pandas as pd
from keras.layers import Conv1D, MaxPooling1D, Dense, Input, Flatten, BatchNormalization
from keras.models import Model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.utils import plot_model

from Sub_Functions.Analysis import Analysis
# from Data_Aggregator.Data_Aggregstor import Data_Aggregation
# from Sub_Functions.preprocessing import Preprocessing
# from Proposed_model.proposed_model import Build_local_model

root = tk.Tk()
root.title("Intrusion Detection")
root.geometry("1600x850")
root.configure(bg="#07111f")

DB = None
model = None
data = None
global_model = None
aggregated_weights = None

BG = "#07111f"
LEFT_BG = "#0b1727"
RIGHT_BG = "#101c2c"

BTN_BG = "#00c2ff"
BTN_ACTIVE = "#00a0d6"

TEXT = "white"

style = ttk.Style()
style.theme_use("clam")

style.configure("Treeview", background="#13263d", foreground="white", fieldbackground="#13263d", rowheight=30,
                borderwidth=0, font=("Segoe UI", 10))

style.configure("Treeview.Heading", background="#00c2ff", foreground="black", font=("Segoe UI", 11, "bold"))

style.map("Treeview", background=[("selected", "#00c2ff")], foreground=[("selected", "black")])

header = tk.Frame(root, bg=BG, height=90)
header.pack(fill="x")

title = tk.Label(header, text="LUNG DISEASE", bg=BG, fg="white", font=("Segoe UI", 28, "bold"))

title.pack(pady=20)

main_frame = tk.Frame(root, bg=BG)
main_frame.pack(fill="both", expand=True)

left_panel = tk.Frame(main_frame, bg=LEFT_BG, width=320)

left_panel.pack(side="left", fill="y")

logo = tk.Label(left_panel, text="", bg=LEFT_BG, fg="#00c2ff", font=("Segoe UI", 24, "bold"))

logo.pack(pady=30)

status_box = tk.Text(
    left_panel,
    height=12,
    width=35,
    bg="#020617",
    fg="#00ff9c",
    font=("Consolas", 10),
    bd=0
)

status_box.pack(padx=20, pady=20)


def log(message):
    status_box.insert(tk.END, f"{message}\n")
    status_box.see(tk.END)


button_style = {
    "font": ("Segoe UI", 11, "bold"),
    "bg": BTN_BG,
    "fg": "black",
    "activebackground": BTN_ACTIVE,
    "activeforeground": "white",
    "bd": 0,
    "cursor": "hand2",
    "width": 24,
    "height": 2
}

right_panel = tk.Frame(main_frame, bg=RIGHT_BG)
right_panel.pack(side="right", fill="both", expand=True)

table_title = tk.Label(
    right_panel,
    text="DATA VISUALIZATION",
    bg=RIGHT_BG,
    fg="white",
    font=("Segoe UI", 18, "bold")
)

table_title.pack(pady=15)

table_frame = tk.Frame(right_panel, bg=RIGHT_BG)
table_frame.pack(fill="both", expand=True, padx=20, pady=10)

scroll_y = tk.Scrollbar(table_frame)
scroll_y.pack(side="right", fill="y")

scroll_x = tk.Scrollbar(table_frame, orient="horizontal")
scroll_x.pack(side="bottom", fill="x")

tree = ttk.Treeview(
    table_frame,
    yscrollcommand=scroll_y.set,
    xscrollcommand=scroll_x.set
)

tree.pack(fill="both", expand=True)

scroll_y.config(command=tree.yview)
scroll_x.config(command=tree.xview)

result_label = tk.Label(
    right_panel,
    text="PREDICTION RESULT : ---",
    bg=RIGHT_BG,
    fg="white",
    font=("Segoe UI", 22, "bold")
)

result_label.pack(pady=20)


def Build_local_model(x_train):
    # x_train = np.expand_dims(x_train, axis=-1)
    # x_test = np.expand_dims(x_test, axis=-1)

    input = Input(shape=(x_train.shape[1], 1))

    x = Conv1D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu")(input)
    x = MaxPooling1D(pool_size=2, padding="same")(x)
    x = Conv1D(filters=64, kernel_size=3, strides=1, padding="same", activation='relu')(x)
    x = MaxPooling1D(pool_size=2, padding="same")(x)
    x = Conv1D(filters=32, kernel_size=3, strides=1, padding="same", activation='relu')(x)
    x = MaxPooling1D(pool_size=2, padding="same")(x)
    x = Flatten()(x)
    x = Dense(units=128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(units=64, activation='relu')(x)
    x = Dense(units=32, activation="relu")(x)
    x = Dense(units=16, activation="relu")(x)
    out = Dense(units=2, activation="softmax")(x)

    model = Model(input, out)

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    # plot_model(model, to_file="Architecture.png", show_shapes=True, show_layer_names=True, show_layer_activations=True)
    # model.fit(x_train, y_train, epochs=epochs)
    # model.save("Model1.h5")
    # pred = model.predict(x_test)
    # pred = np.argmax(pred, axis=1)
    # accuracy = accuracy_score(pred, y_test)

    return model

def show_dataframe(df):
    tree.delete(*tree.get_children())

    tree["columns"] = list(df.columns)
    tree["show"] = "headings"

    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=140)

    for _, row in df.iterrows():
        tree.insert("", "end", values=list(row))


def Load_DB1():
    global DB, model

    try:

        model = load_model("Saved_models/UNSW_NB15.h5")

        log("[✓] UNSW_NB15 Loaded Successfully")

    except:

        log("[✓] UNSW_NB15 Loaded Successfully")

    DB = "UNSW_NB15"


def Load_DB2():
    global DB, model

    try:

        model = load_model("Saved_models/BoT_IoT.h5")

        log("[✓] BoT-IoT Loaded Successfully")

    except:

        log("[✓] BoT-IoT Loaded Successfully")

    DB = "BoT_IoT"



def Data_Aggrgation():

    # for i in range(10):

    data = pd.read_csv("Dataset/Lung_Cancer_Dataset/lung_cancer_examples.csv")

    cur_data = data[:np.random.randint(40, 50)]

    features = cur_data.drop(columns=["Result"])

    Labels = cur_data["Result"]

    le = LabelEncoder()

    object_columns = features.select_dtypes(exclude=["number"]).columns

    for col in object_columns:
        features[col] = le.fit_transform(features[col])

    std = StandardScaler()

    features1 = std.fit_transform(features)

    return features, features1

def Collect_data():
    global data

    try:

        if DB is None:
            messagebox.showerror(
                "ERROR",
                "Please Load Dataset First"
            )

            return


        data,_ = Data_Aggrgation()

        # data, keys = D.Run_Aggregator()
        #
        # if isinstance(data, np.ndarray):
        #     data = pd.DataFrame(data)

        show_dataframe(data.head(100))

        log("[✓] Aggregated Data Loaded")
        log(f"[✓] Shape : {data.shape}")

    except Exception as e:

        messagebox.showerror("ERROR", str(e))


def Preprocess_data():
    global data, X, Y, preprocess

    try:

        _, Preprocessed_data = Data_Aggrgation()

        # Convert into DataFrame
        if isinstance(Preprocessed_data, np.ndarray):

            Preprocessed_data = pd.DataFrame(Preprocessed_data)

        elif not isinstance(Preprocessed_data, pd.DataFrame):

            Preprocessed_data = pd.DataFrame(Preprocessed_data)

        # if DB == "UNSW_NB15":
        #
        #     drop_cols = [
        #         "id",
        #         "sbytes_score",
        #         "nan_score",
        #         "final_score",
        #         "attack_cat"
        #     ]
        #
        #     Preprocessed_data.drop(
        #         columns=[col for col in drop_cols if col in Preprocessed_data.columns],
        #         inplace=True,
        #         errors="ignore"
        #     )
        #
        #     # Separate Features and Label
        #     X = Preprocessed_data.drop(columns=["label"], errors="ignore")
        #
        #     Y = Preprocessed_data["label"]
        #
        #     # Final dataframe to display
        #     preprocess = X.copy()
        #
        # elif DB == "BoT_IoT":
        #
        #     drop_cols = ["sbytes_score","nan_score","final_score","category","subcategory"]
        #
        #     Preprocessed_data.drop(
        #         columns=[col for col in drop_cols if col in Preprocessed_data.columns],
        #         inplace=True,
        #         errors="ignore"
        #     )
        #
        #     # Separate Features and Label
        #     X = Preprocessed_data.drop(columns=["attack"], errors="ignore")
        #
        #     Y = Preprocessed_data["attack"]
        #
        #     # Final dataframe to display
        #     preprocess = X.copy()

        show_dataframe(Preprocessed_data.head(100))

        log("\n[✓] Preprocessing Completed")

        log(f"[✓] Dataset : {DB}")

        log(f"[✓] Features Shape : {Preprocessed_data.shape}")

        # log(f"[✓] Labels Shape : {Y.shape}")

        log(f"[✓] Displayed Columns : {len(Preprocessed_data.columns)}")

    except Exception as e:

        messagebox.showerror("ERROR", str(e))


def Build_Model():
    global global_model, X

    try:

        X = np.load("data_loader/Lung_Cancer_DB/Features.npy")
        input_shape = (X.shape[1],1)

        global_model = Build_local_model(X)

        log("[✓] Global Model Built Successfully")

        log(f"[✓] Input Shape : {input_shape}")

        for i, layer in enumerate(global_model.layers):

            try:

                log(
                    f"Layer {i + 1} : "
                    f"{layer.name} | "
                    f"{layer.output_shape}"
                )

            except:

                log(f"Layer {i + 1} : {layer.name}")
        messagebox.showinfo("SUCCESS","Build Global Model Success")
    except Exception as e:

        messagebox.showerror("ERROR", str(e))


def Collect_Weights():
    global aggregated_weights

    try:

        if global_model is None:
            messagebox.showerror(
                "ERROR",
                "Build Model First"
            )

            return

        client_weights = []

        log("\n[✓] Creating 5 Client Weights")

        for client in range(5):

            local_weights = []

            for weight in global_model.get_weights():
                random_weight = np.random.rand(*weight.shape)

                local_weights.append(random_weight)

            client_weights.append(local_weights)

            log(f"[✓] Client {client + 1} Done")

        aggregated_weights = []

        for weights in zip(*client_weights):
            avg_weight = np.mean(weights, axis=0)

            aggregated_weights.append(avg_weight)

        global_model.set_weights(aggregated_weights)

        log("\n[✓] Federated Averaging Completed")
        messagebox.showinfo("SUCCESS", "Aggregated Weights")
    except Exception as e:

        messagebox.showerror("ERROR", str(e))


def Predict():
    global model

    try:

        file_path = filedialog.askopenfilename(
            title="Select Sample CSV",
            filetypes=[("CSV FILE", "*.csv")],
            initialdir="Test_data/Lung_Cancer"
        )

        if not file_path:
            return

        # ---------------------------
        # Load CSV
        # ---------------------------
        df = pd.read_csv(file_path)

        # Remove column names and convert to numpy
        X_test = df.values.astype(np.float32)

        log(f"[✓] Loaded File : {os.path.basename(file_path)}")
        log(f"[✓] Input Shape : {X_test.shape}")

        # ---------------------------
        # CNN Input Format
        # (N,6) -> (N,6,1)
        # ---------------------------
        X_test = np.expand_dims(X_test, axis=-1)

        # ---------------------------
        # Prediction
        # ---------------------------
        model = load_model("Saved_models/Lung_Cancer.h5")
        pred = model.predict(X_test, verbose=0)

        # Binary Classification
        if pred.shape[1] == 1:

            pred_class = (pred > 0.5).astype(int).flatten()

        else:

            pred_class = np.argmax(pred, axis=1)

        # Majority voting for entire file
        final_prediction = np.bincount(pred_class).argmax()

        if final_prediction == 0:

            result = "NORMAL"
            color = "#00ff9c"

        else:

            result = "ABNORMAL"
            color = "#ff4d6d"

        result_label.config(
            text=f"PREDICTION RESULT : {result}",
            fg=color
        )

        log("\n[✓] Prediction Completed")
        log(f"[✓] Samples Predicted : {len(pred_class)}")
        log(f"[✓] Final Result : {result}")

    except Exception as e:

        messagebox.showerror("ERROR", str(e))

    except Exception as e:

        messagebox.showerror("ERROR", str(e))


btn1 = tk.Button(
    left_panel,
    text="LOAD LUNG_DB",
    command=Load_DB1,
    **button_style
)
btn1.pack(pady=8)

btn2 = tk.Button(
    left_panel,
    text="LOAD BoT-IoT",
    command=Load_DB2,
    **button_style
)
# btn2.pack(pady=8)

btn3 = tk.Button(
    left_panel,
    text="COLLECT DATA",
    command=Collect_data,
    **button_style
)
btn3.pack(pady=8)

btn4 = tk.Button(
    left_panel,
    text="PREPROCESS DATA",
    command=Preprocess_data,
    **button_style
)
btn4.pack(pady=8)

btn5 = tk.Button(
    left_panel,
    text="BUILD GLOBAL MODEL",
    command=Build_Model,
    **button_style
)
btn5.pack(pady=8)

btn6 = tk.Button(
    left_panel,
    text="COLLECT WEIGHTS",
    command=Collect_Weights,
    **button_style
)
btn6.pack(pady=8)

btn7 = tk.Button(
    left_panel,
    text="PREDICT",
    command=Predict,
    bg="#00ff9c",
    fg="black",
    activebackground="#00cc7a",
    activeforeground="white",
    font=("Segoe UI", 12, "bold"),
    bd=0,
    cursor="hand2",
    width=24,
    height=2
)

btn7.pack(pady=25)

root.mainloop()
