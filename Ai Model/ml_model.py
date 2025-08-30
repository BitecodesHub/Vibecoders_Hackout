"""
Smart Hydrogen Site Optimizer - Machine Learning Component
==========================================================
Complete ML pipeline for predicting optimal hydrogen production sites
with explainable AI features and comprehensive analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('ggplot')
sns.set_palette("husl")

class HydrogenSiteOptimizer:
    """
    Main ML model for hydrogen production site optimization
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.explainer = None
        
    def load_and_prepare_data(self, filepath='renewable_hydrogen_dataset_2535.csv'):
        """
        Load and prepare the hydrogen production dataset
        """
        # Load data
        self.df = pd.read_csv(filepath)
        print(f"✅ Loaded dataset with {len(self.df)} sites and {len(self.df.columns)} features")
        
        # Basic info
        print("\n📊 Dataset Overview:")
        print("="*50)
        print(self.df.info())
        
        print("\n📈 Statistical Summary:")
        print("="*50)
        print(self.df.describe())
        
        # Check for missing values
        missing = self.df.isnull().sum()
        if missing.any():
            print("\n⚠️ Missing values found:")
            print(missing[missing > 0])
        else:
            print("\n✅ No missing values found")
        
        return self.df
    
    def exploratory_data_analysis(self):
        """
        Comprehensive EDA with visualizations
        """
        print("\n🔍 Exploratory Data Analysis")
        print("="*50)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Feasibility Score Distribution
        ax1 = plt.subplot(3, 3, 1)
        self.df['Feasibility_Score'].hist(bins=30, edgecolor='black', alpha=0.7)
        ax1.set_title('Feasibility Score Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Feasibility Score')
        ax1.set_ylabel('Frequency')
        ax1.axvline(self.df['Feasibility_Score'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.df["Feasibility_Score"].mean():.3f}')
        ax1.legend()
        
        # 2. Hydrogen Production Distribution
        ax2 = plt.subplot(3, 3, 2)
        self.df['Hydrogen_Production_kg/day'].hist(bins=30, edgecolor='black', alpha=0.7, color='green')
        ax2.set_title('Hydrogen Production Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Hydrogen Production (kg/day)')
        ax2.set_ylabel('Frequency')
        
        # 3. Geographic Distribution
        ax3 = plt.subplot(3, 3, 3)
        scatter = ax3.scatter(self.df['Longitude'], self.df['Latitude'], 
                            c=self.df['Feasibility_Score'], cmap='viridis', 
                            s=50, alpha=0.6, edgecolor='black', linewidth=0.5)
        ax3.set_title('Geographic Distribution of Sites', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Longitude')
        ax3.set_ylabel('Latitude')
        plt.colorbar(scatter, ax=ax3, label='Feasibility Score')
        
        # 4. Solar vs Wind Power
        ax4 = plt.subplot(3, 3, 4)
        ax4.scatter(self.df['PV_Power_kW'], self.df['Wind_Power_kW'], 
                   c=self.df['Feasibility_Score'], cmap='coolwarm', alpha=0.6)
        ax4.set_title('Solar vs Wind Power Capacity', fontsize=12, fontweight='bold')
        ax4.set_xlabel('PV Power (kW)')
        ax4.set_ylabel('Wind Power (kW)')
        
        # 5. Efficiency Metrics
        ax5 = plt.subplot(3, 3, 5)
        efficiency_cols = ['Electrolyzer_Efficiency_%', 'System_Efficiency_%']
        self.df[efficiency_cols].boxplot(ax=ax5)
        ax5.set_title('Efficiency Metrics Distribution', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Efficiency (%)')
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=15)
        
        # 6. Environmental Factors
        ax6 = plt.subplot(3, 3, 6)
        env_data = self.df[['Solar_Irradiance_kWh/m²/day', 'Wind_Speed_m/s', 'Temperature_C']]
        env_data_normalized = (env_data - env_data.min()) / (env_data.max() - env_data.min())
        env_data_normalized.boxplot(ax=ax6)
        ax6.set_title('Environmental Factors (Normalized)', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Normalized Value')
        plt.setp(ax6.xaxis.get_majorticklabels(), rotation=15)
        
        # 7. Correlation Matrix
        ax7 = plt.subplot(3, 3, 7)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['City']]
        corr_matrix = self.df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                   square=True, ax=ax7, cbar_kws={"shrink": 0.8})
        ax7.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')
        plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.setp(ax7.yaxis.get_majorticklabels(), rotation=0)
        
        # 8. Top Correlations with Feasibility Score
        ax8 = plt.subplot(3, 3, 8)
        correlations = self.df[numeric_cols].corr()['Feasibility_Score'].sort_values(ascending=False)[1:11]
        correlations.plot(kind='barh', ax=ax8, color='teal')
        ax8.set_title('Top 10 Features Correlated with Feasibility', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Correlation Coefficient')
        
        # 9. Production vs Efficiency
        ax9 = plt.subplot(3, 3, 9)
        ax9.scatter(self.df['System_Efficiency_%'], self.df['Hydrogen_Production_kg/day'],
                   c=self.df['Feasibility_Score'], cmap='plasma', alpha=0.6)
        ax9.set_title('Production vs System Efficiency', fontsize=12, fontweight='bold')
        ax9.set_xlabel('System Efficiency (%)')
        ax9.set_ylabel('Hydrogen Production (kg/day)')
        
        plt.suptitle('Hydrogen Site Optimization - Data Analysis Dashboard', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
        
        # Print key statistics
        print("\n📊 Key Statistics:")
        print(f"Average Feasibility Score: {self.df['Feasibility_Score'].mean():.3f}")
        print(f"Average Hydrogen Production: {self.df['Hydrogen_Production_kg/day'].mean():.2f} kg/day")
        print(f"Average System Efficiency: {self.df['System_Efficiency_%'].mean():.2f}%")
        print(f"Best Site Feasibility: {self.df['Feasibility_Score'].max():.3f}")
        print(f"Worst Site Feasibility: {self.df['Feasibility_Score'].min():.3f}")
        
    def feature_engineering(self):
        """
        Create new features to improve model performance
        """
        print("\n🔧 Feature Engineering")
        print("="*50)
        
        # Create a copy for feature engineering
        self.df_features = self.df.copy()
        
        # 1. Total renewable power
        self.df_features['Total_Renewable_Power_kW'] = (
            self.df_features['PV_Power_kW'] + self.df_features['Wind_Power_kW']
        )
        
        # 2. Power mix ratio
        self.df_features['Solar_Wind_Ratio'] = (
            self.df_features['PV_Power_kW'] / 
            (self.df_features['Wind_Power_kW'] + 1)  # Add 1 to avoid division by zero
        )
        
        # 3. Production efficiency
        self.df_features['Production_Efficiency'] = (
            self.df_features['Hydrogen_Production_kg/day'] / 
            (self.df_features['Total_Renewable_Power_kW'] + 1)
        )
        
        # 4. Climate suitability score
        self.df_features['Climate_Score'] = (
            self.df_features['Solar_Irradiance_kWh/m²/day'] * 0.5 +
            self.df_features['Wind_Speed_m/s'] * 0.3 +
            (25 - abs(self.df_features['Temperature_C'] - 25)) * 0.2
        )
        
        # 5. Efficiency composite score
        self.df_features['Efficiency_Composite'] = (
            self.df_features['Electrolyzer_Efficiency_%'] * 0.6 +
            self.df_features['System_Efficiency_%'] * 0.4
        )
        
        # 6. Geographic zones (simplified)
        self.df_features['Lat_Zone'] = pd.cut(self.df_features['Latitude'], 
                                               bins=5, labels=['Zone_1', 'Zone_2', 'Zone_3', 'Zone_4', 'Zone_5'])
        self.df_features['Lon_Zone'] = pd.cut(self.df_features['Longitude'], 
                                               bins=5, labels=['Zone_A', 'Zone_B', 'Zone_C', 'Zone_D', 'Zone_E'])
        
        # 7. Desalination efficiency
        self.df_features['Desalination_Efficiency'] = (
            self.df_features['Desalination_Power_kW'] / 
            self.df_features['Total_Renewable_Power_kW']
        )
        
        print(f"✅ Created {7} new engineered features")
        print("\nNew features:")
        new_features = ['Total_Renewable_Power_kW', 'Solar_Wind_Ratio', 'Production_Efficiency',
                       'Climate_Score', 'Efficiency_Composite', 'Lat_Zone', 'Lon_Zone', 
                       'Desalination_Efficiency']
        for feat in new_features:
            if feat in ['Lat_Zone', 'Lon_Zone']:
                print(f"  - {feat}: Categorical zone feature")
            else:
                print(f"  - {feat}: Mean = {self.df_features[feat].mean():.3f}")
        
        return self.df_features
    
    def prepare_model_data(self):
        """
        Prepare data for model training
        """
        print("\n🎯 Preparing Data for Model Training")
        print("="*50)
        
        # Select features for modeling
        feature_cols = [
            'Latitude', 'Longitude',
            'Solar_Irradiance_kWh/m²/day', 'Temperature_C', 'Wind_Speed_m/s',
            'PV_Power_kW', 'Wind_Power_kW',
            'Electrolyzer_Efficiency_%', 'Hydrogen_Production_kg/day',
            'Desalination_Power_kW', 'System_Efficiency_%',
            'Total_Renewable_Power_kW', 'Solar_Wind_Ratio',
            'Production_Efficiency', 'Climate_Score', 
            'Efficiency_Composite', 'Desalination_Efficiency'
        ]
        
        self.feature_names = feature_cols
        X = self.df_features[feature_cols]
        y = self.df_features['Feasibility_Score']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✅ Training set: {len(self.X_train)} samples")
        print(f"✅ Test set: {len(self.X_test)} samples")
        print(f"✅ Number of features: {len(feature_cols)}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_models(self):
        """
        Train multiple models and compare performance
        """
        print("\n🚀 Training Machine Learning Models")
        print("="*50)
        
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n📈 Training {name}...")
            
            # Train model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_pred_train = model.predict(self.X_train_scaled)
            y_pred_test = model.predict(self.X_test_scaled)
            
            # Calculate metrics
            train_r2 = r2_score(self.y_train, y_pred_train)
            test_r2 = r2_score(self.y_test, y_pred_test)
            train_mse = mean_squared_error(self.y_train, y_pred_train)
            test_mse = mean_squared_error(self.y_test, y_pred_test)
            train_mae = mean_absolute_error(self.y_train, y_pred_train)
            test_mae = mean_absolute_error(self.y_test, y_pred_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                       cv=5, scoring='r2')
            
            results[name] = {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred_test': y_pred_test
            }
            
            print(f"  Train R²: {train_r2:.4f}")
            print(f"  Test R²:  {test_r2:.4f}")
            print(f"  Test MSE: {test_mse:.6f}")
            print(f"  Test MAE: {test_mae:.6f}")
            print(f"  CV R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Select best model
        best_model_name = max(results, key=lambda x: results[x]['test_r2'])
        self.model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   Test R² Score: {results[best_model_name]['test_r2']:.4f}")
        
        # Visualize model comparison
        self._visualize_model_comparison(results)
        
        return results
    
    def _visualize_model_comparison(self, results):
        """
        Visualize model performance comparison
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. R² Score Comparison
        ax1 = axes[0, 0]
        models = list(results.keys())
        train_r2 = [results[m]['train_r2'] for m in models]
        test_r2 = [results[m]['test_r2'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        ax1.bar(x - width/2, train_r2, width, label='Train R²', color='skyblue')
        ax1.bar(x + width/2, test_r2, width, label='Test R²', color='lightcoral')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('R² Score')
        ax1.set_title('Model Performance Comparison (R²)', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=15)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. MSE Comparison
        ax2 = axes[0, 1]
        test_mse = [results[m]['test_mse'] for m in models]
        ax2.bar(models, test_mse, color='orange')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Mean Squared Error')
        ax2.set_title('Test Set MSE Comparison', fontweight='bold')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15)
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Cross-Validation Scores
        ax3 = axes[0, 2]
        cv_means = [results[m]['cv_mean'] for m in models]
        cv_stds = [results[m]['cv_std'] for m in models]
        ax3.errorbar(models, cv_means, yerr=cv_stds, marker='o', capsize=5, capthick=2)
        ax3.set_xlabel('Model')
        ax3.set_ylabel('CV R² Score')
        ax3.set_title('Cross-Validation Performance', fontweight='bold')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=15)
        ax3.grid(alpha=0.3)
        
        # 4-6. Actual vs Predicted for each model
        for idx, (name, res) in enumerate(results.items()):
            ax = axes[1, idx]
            ax.scatter(self.y_test, res['y_pred_test'], alpha=0.5, edgecolor='black', linewidth=0.5)
            ax.plot([self.y_test.min(), self.y_test.max()], 
                   [self.y_test.min(), self.y_test.max()], 
                   'r--', lw=2)
            ax.set_xlabel('Actual Feasibility Score')
            ax.set_ylabel('Predicted Feasibility Score')
            ax.set_title(f'{name}\nActual vs Predicted', fontweight='bold')
            ax.text(0.05, 0.95, f'R² = {res["test_r2"]:.3f}', 
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Model Performance Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
    
    def feature_importance_analysis(self):
        """
        Analyze and visualize feature importance
        """
        print("\n🎯 Feature Importance Analysis")
        print("="*50)
        
        # Get feature importance from the best model
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            # For models without feature_importances_, use permutation importance
            perm_importance = permutation_importance(
                self.model, self.X_test_scaled, self.y_test, n_repeats=10, random_state=42
            )
            importances = perm_importance.importances_mean
        
        # Create importance dataframe
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\n📊 Top 10 Most Important Features:")
        for idx, row in feature_importance_df.head(10).iterrows():
            print(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")
        
        # Visualize feature importance
        plt.figure(figsize=(12, 8))
        
        # Main plot
        plt.subplot(2, 1, 1)
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance_df)))
        bars = plt.bar(range(len(feature_importance_df)), 
                      feature_importance_df['importance'].values,
                      color=colors)
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.title(f'Feature Importance Analysis - {self.best_model_name}', 
                 fontsize=14, fontweight='bold')
        plt.xticks(range(len(feature_importance_df)), 
                  feature_importance_df['feature'].values, 
                  rotation=45, ha='right')
        
        # Top 10 features subplot
        plt.subplot(2, 1, 2)
        top_10 = feature_importance_df.head(10)
        plt.barh(range(len(top_10)), top_10['importance'].values, color='teal')
        plt.yticks(range(len(top_10)), top_10['feature'].values)
        plt.xlabel('Importance Score')
        plt.title('Top 10 Most Important Features', fontsize=12, fontweight='bold')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.show()
        
        return feature_importance_df
    
    def explainable_ai_analysis(self):
        """
        Implement SHAP for model explainability
        """
        print("\n🔍 Explainable AI Analysis with SHAP")
        print("="*50)
        
        # Create SHAP explainer
        if isinstance(self.model, xgb.XGBRegressor):
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = shap.TreeExplainer(self.model)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(self.X_test_scaled)
        
        print("✅ SHAP analysis completed")
        
        # Create separate figures for SHAP plots to avoid conflicts
        # 1. Feature importance bar plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, self.X_test_scaled, 
                         feature_names=self.feature_names, 
                         plot_type="bar", show=False)
        plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # 2. Feature impact summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, self.X_test_scaled, 
                         feature_names=self.feature_names, show=False)
        plt.title('SHAP Feature Impact Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # 3. Dependence plots for top features
        top_features = ['Hydrogen_Production_kg/day', 'System_Efficiency_%', 'Climate_Score']
        
        for feature in top_features:
            if feature in self.feature_names:
                plt.figure(figsize=(8, 5))
                feature_idx = self.feature_names.index(feature)
                shap.dependence_plot(feature_idx, shap_values, self.X_test_scaled,
                                    feature_names=self.feature_names, show=False)
                plt.title(f'SHAP Dependence Plot: {feature}', fontsize=12, fontweight='bold')
                plt.tight_layout()
                plt.show()
        
        # Calculate and display mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        feature_importance_shap = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        print("\n📊 Top Features by SHAP Values:")
        for idx, row in feature_importance_shap.head(10).iterrows():
            print(f"  {row['feature']}: {row['mean_abs_shap']:.4f}")
        
        return shap_values
    
    def predict_site_feasibility(self, latitude, longitude, solar_irradiance=None, 
                                wind_speed=None, temperature=None):
        """
        Predict feasibility for a new site
        """
        # Use average values if environmental factors not provided
        if solar_irradiance is None:
            solar_irradiance = self.df['Solar_Irradiance_kWh/m²/day'].mean()
        if wind_speed is None:
            wind_speed = self.df['Wind_Speed_m/s'].mean()
        if temperature is None:
            temperature = self.df['Temperature_C'].mean()
        
        # Estimate other features based on correlations
        pv_power = 250  # Average estimate
        wind_power = 180  # Average estimate
        electrolyzer_eff = 75  # Average efficiency
        hydrogen_prod = 50  # Average production
        desalination_power = 12  # Average
        system_eff = 78  # Average
        
        # Calculate engineered features
        total_renewable = pv_power + wind_power
        solar_wind_ratio = pv_power / (wind_power + 1)
        production_eff = hydrogen_prod / (total_renewable + 1)
        climate_score = solar_irradiance * 0.5 + wind_speed * 0.3 + (25 - abs(temperature - 25)) * 0.2
        efficiency_composite = electrolyzer_eff * 0.6 + system_eff * 0.4
        desalination_eff = desalination_power / total_renewable
        
        # Create feature vector
        features = np.array([[
            latitude, longitude, solar_irradiance, temperature, wind_speed,
            pv_power, wind_power, electrolyzer_eff, hydrogen_prod,
            desalination_power, system_eff, total_renewable, solar_wind_ratio,
            production_eff, climate_score, efficiency_composite, desalination_eff
        ]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        feasibility_score = self.model.predict(features_scaled)[0]
        
        # Get feature contributions using SHAP
        if self.explainer:
            shap_values = self.explainer.shap_values(features_scaled)
            
            # Create explanation dictionary
            explanation = {
                'feature': self.feature_names,
                'value': features[0],
                'shap_contribution': shap_values[0]
            }
            explanation_df = pd.DataFrame(explanation).sort_values('shap_contribution', 
                                                                  ascending=False)
        else:
            explanation_df = None
        
        return {
            'feasibility_score': feasibility_score,
            'classification': self._classify_score(feasibility_score),
            'explanation': explanation_df,
            'recommendations': self._generate_recommendations(feasibility_score, explanation_df)
        }
    
    def _classify_score(self, score):
        """
        Classify feasibility score into categories
        """
        if score >= 0.95:
            return "Excellent - Highly Recommended"
        elif score >= 0.90:
            return "Very Good - Recommended"
        elif score >= 0.85:
            return "Good - Viable"
        elif score >= 0.80:
            return "Fair - Consider with improvements"
        else:
            return "Poor - Not Recommended"
    
    def _generate_recommendations(self, score, explanation_df):
        """
        Generate site-specific recommendations
        """
        recommendations = []
        
        if score >= 0.90:
            recommendations.append("✅ This site shows excellent potential for hydrogen production")
            recommendations.append("📈 Consider large-scale investment with current configuration")
        elif score >= 0.85:
            recommendations.append("👍 This site is viable for hydrogen production")
            recommendations.append("💡 Minor optimizations could improve feasibility")
        else:
            recommendations.append("⚠️ This site needs significant improvements")
            
            if explanation_df is not None:
                # Find areas for improvement
                bottom_contributors = explanation_df.nsmallest(3, 'shap_contribution')
                for _, row in bottom_contributors.iterrows():
                    if 'Efficiency' in row['feature']:
                        recommendations.append("🔧 Improve system efficiency components")
                    elif 'Power' in row['feature']:
                        recommendations.append("⚡ Increase renewable power capacity")
                    elif 'Climate' in row['feature'] or 'Wind' in row['feature'] or 'Solar' in row['feature']:
                        recommendations.append("🌍 Consider alternative locations with better climate conditions")
        
        return recommendations
    
    def find_optimal_sites(self, n_sites=10):
        """
        Find the top N optimal sites from the dataset
        """
        print(f"\n🏆 Top {n_sites} Optimal Hydrogen Production Sites")
        print("="*50)
        
        # Get predictions for all sites
        all_predictions = self.model.predict(self.scaler.transform(self.df_features[self.feature_names]))
        
        # Create results dataframe
        results_df = self.df_features.copy()
        results_df['Predicted_Feasibility'] = all_predictions
        results_df['Feasibility_Difference'] = results_df['Predicted_Feasibility'] - results_df['Feasibility_Score']
        
        # Sort by predicted feasibility
        top_sites = results_df.nlargest(n_sites, 'Predicted_Feasibility')
        
        print("\n📍 Optimal Site Rankings:\n")
        for idx, (_, site) in enumerate(top_sites.iterrows(), 1):
            print(f"{idx}. {site['City']}")
            print(f"   📍 Location: ({site['Latitude']:.2f}, {site['Longitude']:.2f})")
            print(f"   🎯 Predicted Feasibility: {site['Predicted_Feasibility']:.4f}")
            print(f"   ⚡ Hydrogen Production: {site['Hydrogen_Production_kg/day']:.1f} kg/day")
            print(f"   ☀️ Solar Power: {site['PV_Power_kW']:.0f} kW | 💨 Wind Power: {site['Wind_Power_kW']:.0f} kW")
            print(f"   🔧 System Efficiency: {site['System_Efficiency_%']:.1f}%")
            print(f"   🌡️ Climate Score: {site['Climate_Score']:.2f}")
            print()
        
        # Visualize top sites
        self._visualize_top_sites(top_sites)
        
        return top_sites
    
    def _visualize_top_sites(self, top_sites):
        """
        Visualize characteristics of top sites
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Geographic distribution of top sites
        ax1 = axes[0, 0]
        scatter = ax1.scatter(self.df_features['Longitude'], self.df_features['Latitude'],
                            c='lightgray', s=20, alpha=0.5, label='All Sites')
        ax1.scatter(top_sites['Longitude'], top_sites['Latitude'],
                   c=top_sites['Predicted_Feasibility'], cmap='viridis',
                   s=100, edgecolor='red', linewidth=2, label='Top Sites')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title('Geographic Distribution of Optimal Sites', fontweight='bold')
        ax1.legend()
        
        # 2. Production capacity comparison
        ax2 = axes[0, 1]
        sites_names = [f"Site {i+1}" for i in range(min(10, len(top_sites)))]
        x = np.arange(len(sites_names))
        width = 0.35
        ax2.bar(x - width/2, top_sites['PV_Power_kW'].values[:len(sites_names)], 
               width, label='Solar Power', color='gold')
        ax2.bar(x + width/2, top_sites['Wind_Power_kW'].values[:len(sites_names)], 
               width, label='Wind Power', color='skyblue')
        ax2.set_xlabel('Site Rank')
        ax2.set_ylabel('Power Capacity (kW)')
        ax2.set_title('Renewable Power Capacity of Top Sites', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(sites_names, rotation=45)
        ax2.legend()
        
        # 3. Efficiency metrics - Fixed boxplot
        ax3 = axes[1, 0]
        # Prepare data for boxplot - each column should be a separate box
        electrolyzer_data = top_sites['Electrolyzer_Efficiency_%'].values[:10]
        system_data = top_sites['System_Efficiency_%'].values[:10]
        efficiency_data = [electrolyzer_data, system_data]
        
        ax3.boxplot(efficiency_data, labels=['Electrolyzer', 'System'])
        ax3.set_ylabel('Efficiency (%)')
        ax3.set_title('Efficiency Distribution of Top Sites', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Hydrogen production vs feasibility
        ax4 = axes[1, 1]
        ax4.scatter(top_sites['Hydrogen_Production_kg/day'], 
                   top_sites['Predicted_Feasibility'],
                   c=range(len(top_sites)), cmap='coolwarm', s=100, alpha=0.7)
        ax4.set_xlabel('Hydrogen Production (kg/day)')
        ax4.set_ylabel('Predicted Feasibility Score')
        ax4.set_title('Production vs Feasibility for Top Sites', fontweight='bold')
        ax4.grid(alpha=0.3)
        
        # Add rank labels
        for i, (_, site) in enumerate(top_sites.head(5).iterrows()):
            ax4.annotate(f"#{i+1}", 
                        (site['Hydrogen_Production_kg/day'], site['Predicted_Feasibility']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.suptitle('Analysis of Top Hydrogen Production Sites', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()


def run_complete_analysis():
    """
    Run the complete hydrogen site optimization analysis
    """
    print("="*60)
    print("🌊 SMART HYDROGEN SITE OPTIMIZER 🌊")
    print("Machine Learning Model Development & Analysis")
    print("="*60)
    
    # Initialize the optimizer
    optimizer = HydrogenSiteOptimizer()
    
    # Step 1: Load and prepare data
    print("\n📂 Step 1: Loading Data")
    df = optimizer.load_and_prepare_data('renewable_hydrogen_dataset_2535.csv')
    
    # Step 2: Exploratory Data Analysis
    print("\n📊 Step 2: Exploratory Data Analysis")
    optimizer.exploratory_data_analysis()
    
    # Step 3: Feature Engineering
    print("\n🔧 Step 3: Feature Engineering")
    df_features = optimizer.feature_engineering()
    
    # Step 4: Prepare model data
    print("\n🎯 Step 4: Data Preparation")
    X_train, X_test, y_train, y_test = optimizer.prepare_model_data()
    
    # Step 5: Train models
    print("\n🚀 Step 5: Model Training")
    results = optimizer.train_models()
    
    # Step 6: Feature importance analysis
    print("\n📈 Step 6: Feature Importance")
    feature_importance = optimizer.feature_importance_analysis()
    
    # Step 7: Explainable AI
    print("\n🔍 Step 7: Explainable AI")
    shap_values = optimizer.explainable_ai_analysis()
    
    # Step 8: Find optimal sites
    print("\n🏆 Step 8: Optimal Site Identification")
    top_sites = optimizer.find_optimal_sites(n_sites=10)
    
    # Step 9: Example prediction for a new site
    print("\n🎯 Step 9: Sample Prediction for New Site")
    print("="*50)
    
    # Example: Predict for a location near Ahmedabad, Gujarat, India
    sample_prediction = optimizer.predict_site_feasibility(
        latitude=23.0225,
        longitude=72.5714,
        solar_irradiance=5.5,
        wind_speed=3.2,
        temperature=28.5
    )
    
    print(f"📍 Location: Ahmedabad, Gujarat (23.02°N, 72.57°E)")
    print(f"🎯 Predicted Feasibility Score: {sample_prediction['feasibility_score']:.4f}")
    print(f"📊 Classification: {sample_prediction['classification']}")
    print("\n💡 Recommendations:")
    for rec in sample_prediction['recommendations']:
        print(f"   {rec}")
    
    if sample_prediction['explanation'] is not None:
        print("\n🔍 Top Contributing Factors:")
        for _, row in sample_prediction['explanation'].head(5).iterrows():
            contribution = "positive" if row['shap_contribution'] > 0 else "negative"
            print(f"   - {row['feature']}: {contribution} impact ({row['shap_contribution']:.4f})")
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE!")
    print("="*60)
    
    return optimizer


# Run the complete analysis
if __name__ == "__main__":
    optimizer = run_complete_analysis()
    
    print("\n📌 Model is ready for deployment!")
    print("The trained model can now be integrated with the Spring Boot backend")
    print("and used for real-time predictions through the API.")