import mysql.connector
import io

class Database:
    def __init__(self, host, user, password, database):
        self.host       = host
        self.user       = user
        self.password   = password
        self.database   = database
        self.connection = None
        self.cursor     = None

    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            self.cursor = self.connection.cursor()
            print("Connected to MySQL successfully!")
        except mysql.connector.Error as e:
            print("Error connecting to MySQL:", e)

    def reconnect(self):
        if self.connection:
            self.connection.close()
        self.connect()

    def create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS forensic_samples (
            id          INT AUTO_INCREMENT PRIMARY KEY,
            image_blob  LONGBLOB      NOT NULL,
            species     VARCHAR(100)  NOT NULL,
            inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.cursor.execute(query)
        self.connection.commit()
        print("Table ready.")

    def insert_photo(self, photo_data, species):
        """Insert one image with its class label."""
        query = "INSERT INTO forensic_samples (image_blob, species) VALUES (%s, %s)"
        self.cursor.execute(query, (photo_data, species))
        self.connection.commit()
        print(f"Inserted image as '{species}'")

    def fetch_all_labelled(self):
        """Return list of (image_bytes, label_string) tuples."""
        query = "SELECT image_blob, species FROM forensic_samples"
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def insert_evidence_p2(self, photo_data, evidence_type, species, class_label):
        """Phase 2 insert — stores species + class + evidence type together."""
        query = """
            INSERT INTO evidence_samples_p2
                (image_blob, evidence_type, species, class_label)
            VALUES (%s, %s, %s, %s)
        """
        self.cursor.execute(query, (photo_data, evidence_type,
                                    species, class_label))
        self.connection.commit()

    def fetch_by_evidence_type(self, evidence_type):
        """Returns all (image_bytes, species, class_label) for one evidence type."""
        query = """
            SELECT image_blob, species, class_label
            FROM evidence_samples_p2
            WHERE evidence_type = %s
        """
        self.cursor.execute(query, (evidence_type,))
        return self.cursor.fetchall()

    def fetch_p2_summary(self):
        """Returns count per species per evidence type — for database view panel."""
        query = """
            SELECT evidence_type, species, class_label, COUNT(*) as count
            FROM evidence_samples_p2
            GROUP BY evidence_type, species, class_label
            ORDER BY evidence_type, class_label, species
        """
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def create_table_p2(self):
        """Creates Phase 2 table if it doesn't exist."""
        query = """
        CREATE TABLE IF NOT EXISTS evidence_samples_p2 (
            id             INT AUTO_INCREMENT PRIMARY KEY,
            image_blob     LONGBLOB       NOT NULL,
            evidence_type  VARCHAR(50)    NOT NULL,
            species        VARCHAR(100)   NOT NULL,
            class_label    VARCHAR(100)   NOT NULL,
            inserted_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.cursor.execute(query)
        self.connection.commit()
        print("Phase 2 table ready.")

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        print("MySQL connection closed.")