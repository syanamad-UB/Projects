DROP TABLE IF EXISTS client_loan_details;
DROP TABLE IF EXISTS client_socioeconomic_details;
DROP TABLE IF EXISTS client_details;
DROP TABLE IF EXISTS client_lastcontact_details;
DROP TABLE IF EXISTS client_job_details;
DROP TABLE IF EXISTS client_campaign_details;


CREATE TABLE client_campaign_details(
	client_id numeric(10,0) Primary Key,
	campaign numeric(3,0),
	pdays numeric(4,0),
	previous numeric(3,0),
	poutcome varchar(12),
	outcome varchar(3)
);

CREATE TABLE client_lastcontact_details(
	phone_num varchar(20) primary key,
	client_id numeric(10,0),
	contact varchar(10),
	day_of_week varchar(10),
	month varchar(10),
	duration numeric(5,0),
	foreign key(client_id) references client_campaign_details(client_id) on update cascade on delete cascade
);

CREATE TABLE client_job_details(
	job_id varchar(6) Primary Key,
	client_id numeric(10,0),
	job varchar(20),
	salary numeric(15),
	foreign key(client_id) references client_campaign_details(client_id) on update cascade on delete cascade

);

CREATE TABLE client_details(
	client_id numeric(10,0),
	first_name varchar(100),
	last_name varchar(100),
	city_residence varchar(100),
	age numeric(5,0),
	marital_status varchar(10),
	education varchar(50),
	phone_num varchar(20),
	job_id varchar(10),
	primary key(client_id),
	foreign key(job_id) references client_job_details(job_id) on update cascade on delete cascade,
	foreign key(phone_num) references client_lastcontact_details(phone_num) on update cascade on delete cascade
);
CREATE TABLE client_loan_details(
	loan_id varchar(10) Primary Key,
	client_id numeric(10,0),
	credit_default varchar(7),
	housing_loan varchar(7),
	personal_loan varchar(7),
	credit_score numeric(5,0),
	foreign key(client_id) references client_details(client_id) on update cascade on delete cascade
);

CREATE TABLE client_socioeconomic_details(
	client_id numeric(10,0) Primary Key,
	emp_var_rate numeric(4,2),
	cons_price_idx numeric(5,3),
	cons_conf_idx numeric(5,1),
	euribor_3m numeric(5,2),
	num_of_emp numeric(5,1),
	salary numeric(15),
	job_id varchar(6),
	foreign key(job_id) references client_job_details(job_id) on update cascade on delete cascade
);

