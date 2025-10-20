# Source code for data_preprocessing from PLDataModule

    def data_preprocessing(self, data: pd.DataFrame, fit_mode: bool = False) -> np.ndarray:
        """
        Единый метод для всей предобработки данных с новыми правилами:
        1. Удалены ненужные столбцы.
        2. has_metro = False, если underground - NaN.
        3. Пропуски в 'floor' заполняются случайным числом от 1 до 'floors_count'.
        """
        # Создаем копию, чтобы не изменять оригинальный DataFrame
        df = data.copy()

        # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
        # Удаляем бесполезные столбцы с правильным названием 'accommodation_type'
        # Добавляем errors='ignore', чтобы код не падал, если какого-то столбца уже нет
        columns_to_drop = ['commissions', 'deal_type', 'accommodation_type']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

        # --- Умное заполнение пропусков в 'floor' ---
        def fill_random_floor(row):
            if pd.isna(row['floor']) and pd.notna(row['floors_count']) and row['floors_count'] > 0:
                return np.random.randint(1, int(row['floors_count']) + 1)
            return row['floor']
        df['floor'] = df.apply(fill_random_floor, axis=1)

        # === 1. Feature Engineering ===
        # --- Улучшенная логика для 'has_metro' ---
        metro_cities = ["Москва", "Казань", "Санкт-Петербург"]
        df["has_metro"] = df["location"].apply(lambda x: 1 if x in metro_cities else 0).astype(float)
        df.loc[data['underground'].isna(), 'has_metro'] = 0.0

        df['floor_ratio'] = (df['floor'] / (df['floors_count'] + 1e-6)).astype(float)
        df['meters_per_room'] = (df['total_meters'] / (df['rooms_count'].replace(0, 1) + 1e-6)).astype(float)

        # === 2. Стандартная обработка пропусков ===
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        if 'ID' in numerical_cols: numerical_cols.remove('ID')
        if 'ID' in categorical_cols: categorical_cols.remove('ID')

        if fit_mode:
            self.numerical_imputer = SimpleImputer(strategy='median')
            self.categorical_imputer = SimpleImputer(strategy='most_frequent')
            df[numerical_cols] = self.numerical_imputer.fit_transform(df[numerical_cols])
            df[categorical_cols] = self.categorical_imputer.fit_transform(df[categorical_cols])
        else:
            if not all([self.numerical_imputer, self.categorical_imputer]):
                raise RuntimeError("Imputers have not been fitted. Run with fit_mode=True first.")
            df[numerical_cols] = self.numerical_imputer.transform(df[numerical_cols])
            df[categorical_cols] = self.categorical_imputer.transform(df[categorical_cols])

        # === 3. Top-K One-Hot Encoding ===
        new_ohe_features = []
        ohe_column_names = []
        for col, k in self.top_k_config.items():
            if col in df.columns:
                if fit_mode:
                    top_k = df[col].value_counts().nlargest(k).index.tolist()
                    self.top_k_values[col] = top_k
                for category in self.top_k_values.get(col, []):
                    new_ohe_features.append((df[col] == category).astype(int))
                    ohe_column_names.append(f"{col}_{category}")
        if new_ohe_features:
            ohe_df = pd.concat(new_ohe_features, axis=1)
            ohe_df.columns = ohe_column_names
            df = pd.concat([df.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)
        df = df.drop(columns=list(self.top_k_config.keys()), errors='ignore')

        # === 4. Финальная обработка и масштабирование ===
        df = pd.get_dummies(df, dummy_na=True, dtype=int)
        if 'ID' in df.columns:
            df = df.drop(columns=['ID'])

        if fit_mode:
            self.final_columns = df.columns.tolist()
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(df)
        else:
            df = df.reindex(columns=self.final_columns, fill_value=0)
            scaled_data = self.scaler.transform(df)

        return scaled_data.astype(float)


# Source code for save_preprocessing from PLDataModule

    def save_preprocessing(self, directory_path: str) -> Optional[str]:
        """
        Сохраняет ВСЕ артефакты, необходимые для воспроизведения предобработки.
        """
        self.preprocessing_artifacts = {
            'scaler': self.scaler,
            'numerical_imputer': self.numerical_imputer,
            'categorical_imputer': self.categorical_imputer,
            'final_columns': self.final_columns,
            'top_k_values': self.top_k_values,
            'top_k_config': self.top_k_config # Сохраняем и саму конфигурацию
        }
        if not all(self.preprocessing_artifacts.values()):
            print("Предупреждение: Не все артефакты предобработки были созданы. Сохранение отменено.")
            return None
        preprocess_dir = os.path.join(directory_path, "preprocessing")
        os.makedirs(preprocess_dir, exist_ok=True)
        artifacts_path = os.path.join(preprocess_dir, "preprocessing_artifacts.joblib")
        try:
            joblib.dump(self.preprocessing_artifacts, artifacts_path)
            print(f"Артефакты предобработки успешно сохранены в: {artifacts_path}")
            return preprocess_dir
        except Exception as e:
            print(f"Ошибка при сохранении артефактов предобработки: {e}")
            return None
