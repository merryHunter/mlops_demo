-- Create predictions table with partitioning
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL,
    date TIMESTAMP NOT NULL,
    customerId VARCHAR(50) NOT NULL,
    feature1 FLOAT,
    feature2 FLOAT,
    feature1_minus_feature2 FLOAT,
    feature1_times_feature2 FLOAT,
    predictions FLOAT,
    target FLOAT,
    model_name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, customerId, model_name)
);