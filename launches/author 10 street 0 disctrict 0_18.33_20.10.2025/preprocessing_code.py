# Source code for data_preprocessing from PLDataModule

    def data_preprocessing(self, data: pd.DataFrame, fit_mode: bool = False) -> np.ndarray:
        """
        Единый метод для всей предобработки данных с продвинутым feature engineering:
        1. Создание бинов для total_meters.
        2. Создание признака is_studio для rooms_count <= 0.
        3. Создание сложного категориального признака metro_type.
        """
        df = data.copy()

        # --- ШАГ 1: Начальная очистка и Feature Engineering ---

        # Удаляем ненужные столбцы
        columns_to_drop = ['commissions', 'deal_type', 'accommodation_type']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

        # НОВОЕ: Обработка rooms_count <= 0 как студий
        df['is_studio'] = (df['rooms_count'] <= 0).astype(int)
        # Заменяем некорректные значения на 1, чтобы не мешать дальнейшим вычислениям
        df['rooms_count'] = df['rooms_count'].apply(lambda x: 1 if x <= 0 else x)

        # НОВОЕ: Создание категорий для total_meters (бинаризация)
        # Границы бинов можно подбирать, основываясь на графике распределения
        bins = [0, 30, 45, 65, 90, np.inf]
        labels = ['micro', 'small', 'medium', 'large', 'very_large']
        df['total_meters_bin'] = pd.cut(df['total_meters'], bins=bins, labels=labels, right=False)

        # НОВОЕ: Создание сложного признака metro_type
        metro_cities = {"Москва", "Казань", "Санкт-Петербург"} # Используем set для быстрой проверки
        # Можно расширить списки станций для большей точности
        moscow_stations = {"Авиамоторная", "Автозаводская", "Академическая", "Полежаевская"} # и т.д.
        kazan_stations = {"Авиастроительная", "Аметьево", "Горки", "Яшьлек"} # и т.д.
        spb_stations = {"Автово", "Адмиралтейская", "Академическая"} # и т.д.

        def get_metro_type(row):
            # Проверяем, что станция не NaN
            if pd.notna(row['underground']):
                if row['underground'] in moscow_stations: return 'moscow'
                if row['underground'] in kazan_stations:  return 'kazan'
                if row['underground'] in spb_stations:    return 'spb'
            # Если станция NaN, но город с метро, значит "далеко от метро"
            if row['location'] in metro_cities:
                return 'far_from_metro'
            # Во всех остальных случаях - "нет метро в регионе"
            return 'no_metro_in_region'

        df['metro_type'] = df.apply(get_metro_type, axis=1)

        # Умное заполнение пропусков в 'floor'
        def fill_random_floor(row):
            if pd.isna(row['floor']) and pd.notna(row['floors_count']) and row['floors_count'] > 0:
                return np.random.randint(1, int(row['floors_count']) + 1)
            return row['floor']
        df['floor'] = df.apply(fill_random_floor, axis=1)

        # Остальные engineered features
        df['floor_ratio'] = (df['floor'] / (df['floors_count'] + 1e-6)).astype(float)
        df['meters_per_room'] = (df['total_meters'] / (df['rooms_count'].replace(0, 1) + 1e-6)).astype(float)

        # --- ШАГ 2: Стандартная обработка пропусков (теперь после ручного заполнения) ---
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
                raise RuntimeError("Imputers have not been fitted.")
            df[numerical_cols] = self.numerical_imputer.transform(df[numerical_cols])
            df[categorical_cols] = self.categorical_imputer.transform(df[categorical_cols])

        # --- ШАГ 3: Кодирование категориальных признаков ---

        # Top-K кодирование для "длинных" категорий, таких как 'author' или 'street'
        # (остается без изменений)
        new_ohe_features, ohe_column_names = [], []
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

        # One-Hot Encoding для наших новых и других категорий
        # dummy_na=True автоматически создаст колонку для пропущенных значений, если они остались
        df = pd.get_dummies(df, dummy_na=True, dtype=int)
        if 'ID' in df.columns:
            df = df.drop(columns=['ID'])

        # --- ШАГ 4: Финальное выравнивание и масштабирование ---

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
