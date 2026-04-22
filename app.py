# from flask import Flask, request, jsonify, render_template
# import pickle
# import pandas as pd
# import numpy as np

# app = Flask(__name__)

# # Load trained model
# with open("earthquake_damage_model.pkl", "rb") as f:
#     model = pickle.load(f)

# MODEL_COLUMNS = [
#     'district_id', 'count_floors_pre_eq', 'age_building', 'plinth_area_sq_ft',
#     'height_ft_pre_eq', 'land_surface_condition_Flat',
#     'land_surface_condition_Moderate slope', 'land_surface_condition_Steep slope',
#     'foundation_type_Bamboo/Timber', 'foundation_type_Cement-Stone/Brick',
#     'foundation_type_Mud mortar-Stone/Brick', 'foundation_type_Other',
#     'foundation_type_RC', 'roof_type_Bamboo/Timber-Heavy roof',
#     'roof_type_Bamboo/Timber-Light roof', 'roof_type_RCC/RB/RBC',
#     'ground_floor_type_Brick/Stone', 'ground_floor_type_Mud',
#     'ground_floor_type_Other', 'ground_floor_type_RC',
#     'ground_floor_type_Timber', 'other_floor_type_Not applicable',
#     'other_floor_type_RCC/RB/RBC', 'other_floor_type_TImber/Bamboo-Mud',
#     'other_floor_type_Timber-Planck', 'position_Attached-1 side',
#     'position_Attached-2 side', 'position_Attached-3 side',
#     'position_Not attached', 'plan_configuration_Building with Central Courtyard',
#     'plan_configuration_E-shape', 'plan_configuration_H-shape',
#     'plan_configuration_L-shape', 'plan_configuration_Multi-projected',
#     'plan_configuration_Others', 'plan_configuration_Rectangular',
#     'plan_configuration_Square', 'plan_configuration_T-shape',
#     'plan_configuration_U-shape'
# ]

# # IMPORTANT:
# # Replace these feature names with the EXACT column names used while training
# FEATURE_COLUMNS = [
#     "building_age",
#     "floors",
#     "area_sqft",
#     "material_quality",
#     "foundation_score",
#     "seismic_zone",
#     "previous_damage_score",
#     "occupancy_density"
# ]

# # Optional label mapping
# label_map = {
#     0: "Low Risk",
#     1: "Medium Risk",
#     2: "High Risk"
# }
# @app.route("/")
# def home():
#     return render_template("index.html")


# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         data = request.json

#         raw_input = {
#             "district_id": int(data["district_id"]),
#             "count_floors_pre_eq": int(data["count_floors_pre_eq"]),
#             "age_building": int(data["age_building"]),
#             "plinth_area_sq_ft": float(data["plinth_area_sq_ft"]),
#             "height_ft_pre_eq": float(data["height_ft_pre_eq"]),
#             "land_surface_condition": data["land_surface_condition"],
#             "foundation_type": data["foundation_type"],
#             "roof_type": data["roof_type"],
#             "ground_floor_type": data["ground_floor_type"],
#             "other_floor_type": data["other_floor_type"],
#             "position": data["position"],
#             "plan_configuration": data["plan_configuration"]
#         }

#         input_df = pd.DataFrame([raw_input])

#         input_df = pd.get_dummies(input_df)

#         for col in MODEL_COLUMNS:
#             if col not in input_df.columns:
#                 input_df[col] = 0

#         input_df = input_df[MODEL_COLUMNS]

#         prediction = model.predict(input_df)[0]

#         if hasattr(model, "predict_proba"):
#             probabilities = model.predict_proba(input_df)[0]
#             confidence = round(float(np.max(probabilities)) * 100, 2)
#         else:
#             confidence = 80.0

#         result = label_map.get(int(prediction), str(prediction))

#         return jsonify({
#             "risk_level": result,
#             "confidence": confidence
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 400


# @app.route("/predict-batch", methods=["POST"])
# def predict_batch():
#     try:
#         if "file" not in request.files:
#             return jsonify({"error": "No file uploaded"}), 400

#         file = request.files["file"]
#         df = pd.read_csv(file)

#         # Keep only required columns
#         df_model = df[FEATURE_COLUMNS]

#         predictions = model.predict(df_model)

#         df["prediction"] = predictions
#         df["risk_level"] = df["prediction"].map(label_map)

#         return df.to_json(orient="records")

#     except Exception as e:
#         return jsonify({"error": str(e)}), 400


# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load final trained risk model and training columns
model = joblib.load("earthquake_risk_model.pkl")
model_columns = joblib.load("model_columns.pkl")

label_map = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk"
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        raw_input = {
            "district_id": int(data["district_id"]),
            "count_floors_pre_eq": int(data["count_floors_pre_eq"]),
            "age_building": int(data["age_building"]),
            "plinth_area_sq_ft": float(data["plinth_area_sq_ft"]),
            "height_ft_pre_eq": float(data["height_ft_pre_eq"]),
            "land_surface_condition": data["land_surface_condition"],
            "foundation_type": data["foundation_type"],
            "roof_type": data["roof_type"],
            "ground_floor_type": data["ground_floor_type"],
            "other_floor_type": data["other_floor_type"],
            "position": data["position"],
            "plan_configuration": data["plan_configuration"]
        }

        input_df = pd.DataFrame([raw_input])

        # One-hot encode user input
        input_df = pd.get_dummies(input_df)

        # Match training columns exactly
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(input_df)[0]

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            confidence = round(float(np.max(probabilities)) * 100, 2)
        else:
            confidence = None

        result = label_map.get(int(prediction), str(prediction))

        response = {
            "risk_level": result
        }

        if confidence is not None:
            response["confidence"] = confidence

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)