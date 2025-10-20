# Source code for data_preprocessing from PLDataModule

    def data_preprocessing(self, data: pd.DataFrame, y_train: Optional[pd.Series] = None, fit_mode: bool = False) -> np.ndarray:
        """
        Единый метод предобработки с продвинутым feature engineering:
        1. Создание location_rank на основе медианной цены.
        2. Улучшенная категоризация метро с учетом МЦД.
        """
        df = data.copy()

        # --- ШАГ 1: Продвинутый Feature Engineering ---

        # НОВОЕ: Создание location_rank
        if fit_mode:
            if y_train is None:
                raise ValueError("y_train is required for fit_mode to calculate location ranks.")
            
            # Объединяем X_train и y_train, чтобы посчитать медианы
            full_train_df = pd.concat([df, y_train], axis=1)
            
            # Считаем медианную цену по городам и сортируем
            median_price_by_loc = full_train_df.groupby('location')[self.y_label].median().sort_values()
            
            # Создаем карту "город -> ранг"
            self.location_rank_map = {city: rank for rank, city in enumerate(median_price_by_loc.index)}
        
        # Применяем карту для создания нового признака
        df['location_rank'] = df['location'].map(self.location_rank_map)
        
        if fit_mode:
            # Сохраняем медианный ранг для заполнения пропусков в будущем
            self.median_rank = df['location_rank'].median()
        
        # Заполняем пропуски (если в тесте встретится новый город)
        df['location_rank'].fillna(self.median_rank, inplace=True)
        

        # НОВОЕ: Улучшенная категоризация метро с учетом МЦД
        metro_cities = {"Москва", "Казань", "Санкт-Петербург"}
        near_moscow_metro_cities = {"Люберцы", "Подольск", "Долгопрудный", "Реутов", "Кашира"}

        def get_metro_type(row):
            # Если есть станция, это точно город с метро
            if pd.notna(row['underground']):
                if row['location'] == 'Москва': return 'moscow'
                if row['location'] in ('Казань', 'Санкт-Петербург'): return 'kazan_spb'
            
            # Если станции нет, но город из списка с МЦД
            if row['location'] in near_moscow_metro_cities:
                return 'near_moscow'
            
            # Если станции нет, но это Москва/Казань/СПб (значит, далеко от метро)
            if row['location'] in metro_cities:
                return 'far_from_metro'
            
            # Во всех остальных случаях
            return 'no_metro_in_region'

        df['metro_type'] = df.apply(get_metro_type, axis=1)
        
        # Удаляем оригинальный 'location', так как мы его полностью заменили
        df = df.drop(columns=['location'], errors='ignore')


        # --- Остальная часть предобработки (остается как раньше) ---
        columns_to_drop = ['commissions', 'deal_type', 'accommodation_type']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

        df['is_studio'] = (df['rooms_count'] <= 0).astype(int)
        df['rooms_count'] = df['rooms_count'].apply(lambda x: 1 if x <= 0 else x)
        
        bins = [0, 30, 45, 65, 90, np.inf]
        labels = ['micro', 'small', 'medium', 'large', 'very_large']
        df['total_meters_bin'] = pd.cut(df['total_meters'], bins=bins, labels=labels, right=False)
        
        def fill_random_floor(row):
            if pd.isna(row['floor']) and pd.notna(row['floors_count']) and row['floors_count'] > 0:
                return np.random.randint(1, int(row['floors_count']) + 1)
            return row['floor']
        df['floor'] = df.apply(fill_random_floor, axis=1)

        df['floor_ratio'] = (df['floor'] / (df['floors_count'] + 1e-6)).astype(float)
        df['meters_per_room'] = (df['total_meters'] / (df['rooms_count'].replace(0, 1) + 1e-6)).astype(float)

        # --- Стандартная обработка пропусков, кодирование и масштабирование ---
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
