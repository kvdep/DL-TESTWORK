# Source code for data_preprocessing from PLDataModule

    def data_preprocessing(self, data: pd.DataFrame, fit_mode: bool = False) -> np.ndarray:
        """
        Единый метод для всей предобработки данных.
        Если fit_mode=True, обучает импьютеры, скейлер, находит топ-K категории и сохраняет их в self.
        Если fit_mode=False, применяет уже сохраненные в self артефакты.
        """
        df = data.copy()

        # === 1. Feature Engineering ===
        metro_cities = ["Москва", "Казань", "Санкт-Петербург"] # Добавим на всякий случай
        df["has_metro"] = df["location"].apply(lambda x: 1 if x in metro_cities else 0).astype(float)
        df['floor_ratio'] = (df['floor'] / (df['floors_count'] + 1e-6)).astype(float)
        df['meters_per_room'] = (df['total_meters'] / (df['rooms_count'].replace(0, 1) + 1e-6)).astype(float)

        # === 2. Обработка пропусков и типов ===
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        
        # Удаляем ID из списков, если он там есть
        if 'ID' in numerical_cols: numerical_cols.remove('ID')
        if 'ID' in categorical_cols: categorical_cols.remove('ID')

        if fit_mode:
            # Обучаем и сохраняем импьютеры
            self.numerical_imputer = SimpleImputer(strategy='median')
            self.categorical_imputer = SimpleImputer(strategy='most_frequent')
            df[numerical_cols] = self.numerical_imputer.fit_transform(df[numerical_cols])
            df[categorical_cols] = self.categorical_imputer.fit_transform(df[categorical_cols])
        else:
            # Применяем существующие импьютеры
            df[numerical_cols] = self.numerical_imputer.transform(df[numerical_cols])
            df[categorical_cols] = self.categorical_imputer.transform(df[categorical_cols])

        # === 3. Top-K One-Hot Encoding ===
        for col, k in self.top_k_config.items():
            if col in df.columns:
                if fit_mode:
                    # Находим и сохраняем топ-K категорий
                    top_k = df[col].value_counts().nlargest(k).index.tolist()
                    self.top_k_values[col] = top_k
                
                # Применяем One-Hot Encoding для сохраненных категорий
                for category in self.top_k_values[col]:
                    df[f"{col}_{category}"] = (df[col] == category).astype(int)
        
        # Удаляем оригинальные категориальные столбцы, которые были закодированы
        df = df.drop(columns=list(self.top_k_config.keys()), errors='ignore')

        # === 4. Финальная обработка и Dummy-кодирование ===
        # Конвертируем оставшиеся object столбцы в dummy переменные
        df = pd.get_dummies(df, dummy_na=True, dtype=int)
        
        # Удаляем ID, если он все еще есть
        if 'ID' in df.columns:
            df = df.drop(columns=['ID'])

        if fit_mode:
            # Сохраняем финальный список колонок
            self.final_columns = df.columns.tolist()
            # Обучаем и сохраняем скейлер
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(df)
        else:
            # Приводим столбцы в соответствие с обученными
            df = df.reindex(columns=self.final_columns, fill_value=0)
            # Применяем существующий скейлер
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
