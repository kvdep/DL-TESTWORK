    def data_preprocessing(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
        """
        Applies all preprocessing steps from the notebook.
        Fits transformers on X_train and applies them to all splits.
        """
        # --- FIX: Clean invalid 'rooms_count' in ALL datasets first ---
        for df in [X_train, X_val, X_test]:
            df.loc[df["rooms_count"] < 0, "rooms_count"] = np.nan

        # --- Imputation: Fill missing 'location' based on 'underground' ---
        metro_dict = {
            "Москва": ["Авиамоторная", "Автозаводская", "Академическая", "Александровский сад", "Алексеевская", "Алма-Атинская", "Алтуфьево", "Аннино", "Арбатская", "Аэропорт", "Бабушкинская", "Багратионовская", "Баррикадная", "Бауманская", "Беговая", "Беломорская", "Белорусская", "Беляево", "Бибирево", "Библиотека имени Ленина", "Битцевский парк", "Борисово", "Боровицкая", "Боровское шоссе", "Ботанический сад", "Братиславская", "Бульвар адмирала Ушакова", "Бульвар Дмитрия Донского", "Бульвар Рокоссовского", "Бунинская аллея", "Бутырская", "Варшавская", "ВДНХ", "Верхние Лихоборы", "Владыкино", "Водный стадион", "Войковская", "Волгоградский проспект", "Волжская", "Волоколамская", "Воробьевы горы", "Выставочная", "Выхино", "Говорово", "Деловой центр", "Динамо", "Дмитровская", "Добрынинская", "Домодедовская", "Достоевская", "Дубровка", "Жулебино", "Зябликово", "Измайловская", "Калужская", "Кантемировская", "Каховская", "Каширская", "Киевская", "Китай-город", "Кожуховская", "Коломенская", "Коммунарка", "Комсомольская", "Коньково", "Косино", "Котельники", "Красногвардейская", "Краснопресненская", "Красносельская", "Красные ворота", "Крестьянская застава", "Кропоткинская", "Крылатское", "Кузнецкий мост", "Кузьминки", "Кунцевская", "Курская", "Кутузовская", "Ленинский проспект", "Лермонтовский проспект", "Лесопарковая", "Лефортово", "Ломоносовский проспект", "Лубянка", "Лухмановская", "Люблино", "Марксистская", "Марьина роща", "Марьино", "Маяковская", "Медведково", "Международная", "Менделеевская", "Минская", "Митино", "Мичуринский проспект", "Молодежная", "Мякинино", "Нагатинская", "Нагорная", "Нахимовский проспект", "Некрасовка", "Нижегородская", "Новогиреево", "Новокосино", "Новокузнецкая", "Новопеределкино", "Новослободская", "Новоясеневская", "Новые Черемушки", "Озерная", "Окружная", "Окская", "Октябрьская", "Октябрьское поле", "Ольховая", "Орехово", "Отрадное", "Охотный ряд", "Павелецкая", "Парк культуры", "Парк Победы", "Партизанская", "Первомайская", "Перово", "Петровский парк", "Петровско-Разумовская", "Печатники", "Пионерская", "Планерная", "Площадь Ильича", "Площадь Революции", "Полежаевская", "Полянка", "Пражская", "Преображенская площадь", "Прокшино", "Пролетарская", "Проспект Вернадского", "Проспект Мира", "Профсоюзная", "Пушкинская", "Пятницкое шоссе", "Раменки", "Рассказовка", "Речной вокзал", "Рижская", "Римская", "Румянцево", "Рязанский проспект", "Савеловская", "Саларьево", "Свиблово", "Севастопольская", "Селигерская", "Семеновская", "Серпуховская", "Славянский бульвар", "Смоленская", "Сокол", "Сокольники", "Солнцево", "Спартак", "Спортивная", "Сретенский бульвар", "Стахановская", "Строгино", "Студенческая", "Сухаревская", "Сходненская", "Таганская", "Тверская", "Театральная", "Текстильщики", "Теплый Стан", "Технопарк", "Тимирязевская", "Третьяковская", "Тропарево", "Трубная", "Тульская", "Тургеневская", "Тушинская", "Улица 1905 года", "Улица академика Янгеля", "Улица Горчакова", "Улица Дмитриевского", "Улица Скобелевская", "Улица Старокачаловская", "Университет", "Филатов луг", "Филевский парк", "Фили", "Фонвизинская", "Фрунзенская", "Ховрино", "Хорошевская", "Царицыно", "Цветной бульвар", "ЦСКА", "Черкизовская", "Чертановская", "Чеховская", "Чистые пруды", "Чкаловская", "Шаболовская", "Шелепиха", "Шипиловская", "Шоссе Энтузиастов", "Щелковская", "Щукинская", "Электрозаводская", "Юго-Восточная", "Юго-Западная", "Южная", "Ясенево"],
            "Казань": ["Авиастроительная", "Аметьево", "Горки", "Козья слобода", "Кремлёвская", "Площадь Габдуллы Тукая", "Проспект Победы", "Северный вокзал", "Суконная слобода", "Яшьлек", "Юность"]
        }
        for df in [X_train, X_val, X_test]:
            for idx in df[df.location.isna()].index:
                underground_station = df.loc[idx, "underground"]
                if underground_station in metro_dict["Москва"]:
                    df.loc[idx, "location"] = "Москва"
                elif underground_station in metro_dict["Казань"]:
                    df.loc[idx, "location"] = "Казань"
        
        # --- Feature Engineering: Create 'has_metro' ---
        has_metro_cities = ["Москва", "Казань", "Иваново", "Кашира", "Подольск", "Люберцы", "Реутов", "Долгопрудный"]
        for df in [X_train, X_val, X_test]:
            df["has_metro"] = df["location"].apply(lambda x: 1 if x in has_metro_cities else 0)
        self.preprocessing_artifacts['has_metro_cities'] = has_metro_cities

        # --- Imputation: Fill missing 'underground' using modes from training data ---
        metro_data = X_train[X_train['has_metro'] == 1]
        mode_metro_by_district = metro_data.groupby(["location", "district"])['underground'].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
        mode_metro_by_location = metro_data.groupby("location")['underground'].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
        
        mode_metro_by_district = mode_metro_by_district.reset_index()
        mode_metro_by_district['underground'] = mode_metro_by_district['underground'].fillna(
            mode_metro_by_district['location'].map(mode_metro_by_location)
        )
        mode_metro_by_district = mode_metro_by_district.set_index(['location', 'district'])
        self.preprocessing_artifacts['mode_metro_map'] = mode_metro_by_district
        
        for df in [X_train, X_val, X_test]:
            df_merged = df.merge(mode_metro_by_district.reset_index(), on=['location', 'district'], how='left', suffixes=('', '_mode'))
            df['underground'] = df_merged['underground'].fillna(df_merged['underground_mode'])
            df.drop(columns=['underground_mode'], inplace=True, errors='ignore')

        # --- Imputation: Simple fills ---
        for df in [X_train, X_val, X_test]:
            df["deal_type"] = "rent"
            
            floor_na_idx = df[df["floor"].isna()].index
            floors_count_na_idx = df[df["floors_count"].isna()].index
            df.loc[floor_na_idx, "floor"] = df.loc[floor_na_idx, "floors_count"]
            df.loc[floors_count_na_idx, "floors_count"] = df.loc[floors_count_na_idx, "floor"]

        # --- Imputation: 'total_meters' and 'rooms_count' ---
        meters_per_room = X_train.groupby("rooms_count")['total_meters'].median()
        self.preprocessing_artifacts['meters_per_room'] = meters_per_room
        
        for df in [X_train, X_val, X_test]:
            # Impute 'total_meters' based on 'rooms_count'
            df['total_meters'] = df.apply(
                lambda row: meters_per_room.get(row['rooms_count'], row['total_meters']) 
                            if pd.isna(row['total_meters']) else row['total_meters'],
                axis=1
            )
            # Impute 'rooms_count' based on 'total_meters'
            def get_rooms_by_meters(meters):
                if pd.isna(meters): return np.nan
                # Find the closest room count by median meterage
                for i in range(2, 6): # Assuming rooms_count up to 5
                    if meters < meters_per_room.get(i, float('inf')):
                        return i - 1
                return 6 # For very large apartments
            
            df['rooms_count'] = df.apply(
                lambda row: get_rooms_by_meters(row['total_meters']) if pd.isna(row['rooms_count']) else row['rooms_count'],
                axis=1
            )
        
        # --- Feature Selection: Drop unnecessary columns ---
        cols_to_drop = ["author", "author_type", "deal_type", "accommodation_type", "commissions", "district", "street", "house_number", "underground", "ID"]
        for df in [X_train, X_val, X_test]:
            df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True, errors='ignore')

        # --- Label Encoding for 'location' ---
        cities_index = {
            "Москва": 0, "Подольск": 1, "Люберцы": 2, "Долгопрудный": 3, "Реутов": 4, "Казань": 5, "Калуга": 6,
            "Кашира": 7, "Рязань": 8, "Владимир": 9, "Иваново": 10, "Великий Новгород": 11, "Смоленск": 12,
            "Брянск": 13, "Киров": 14
        }
        self.preprocessing_artifacts['cities_index'] = cities_index
        for df in [X_train, X_val, X_test]:
            df["location"] = df["location"].map(cities_index)

        # --- Final list of numerical columns for scaling ---
        numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
        if 'has_metro' in numerical_cols: # has_metro is already binary, no scaling needed
            numerical_cols.remove('has_metro')

        # --- Transformations and Scaling ---
        self.scaler = StandardScaler()
        
        for df in [X_train, X_val, X_test]:
            for col in ["floor", "floors_count", "total_meters"]:
                if col in df.columns:
                    df[col] = np.log(df[col] + 1e-10)
            if "rooms_count" in df.columns:
                df["rooms_count"] = np.sqrt(df["rooms_count"])
        
        # Impute any remaining NaNs after transformations using median from training set
        self.numerical_imputer = SimpleImputer(strategy='median')
        X_train[numerical_cols] = self.numerical_imputer.fit_transform(X_train[numerical_cols])
        X_val[numerical_cols] = self.numerical_imputer.transform(X_val[numerical_cols])
        X_test[numerical_cols] = self.numerical_imputer.transform(X_test[numerical_cols])
        self.preprocessing_artifacts['numerical_imputer'] = self.numerical_imputer

        # Fit scaler on training data and transform all sets
        X_train[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
        X_val[numerical_cols] = self.scaler.transform(X_val[numerical_cols])
        X_test[numerical_cols] = self.scaler.transform(X_test[numerical_cols])
        
        self.preprocessing_artifacts['scaler'] = self.scaler
        self.final_columns = X_train.columns.tolist()
        self.preprocessing_artifacts['final_columns'] = self.final_columns
        
        # Final check for alignment and return numpy arrays
        return X_train.values, X_val[self.final_columns].values, X_test[self.final_columns].values
