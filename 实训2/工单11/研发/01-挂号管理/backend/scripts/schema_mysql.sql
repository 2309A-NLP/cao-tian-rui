-- 工单11 医疗挂号Agent 数据库 Schema (MySQL 8.0)
-- 字符集: utf8mb4，时区: 建议服务器设置 Asia/Shanghai
-- 执行前确保已 USE medical_agent;

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ========== 工单给定表 ==========

CREATE TABLE IF NOT EXISTS department (
    dep_ID       INT NOT NULL AUTO_INCREMENT,
    dep_Name     VARCHAR(50)  NOT NULL,
    dep_Address  VARCHAR(200),
    PRIMARY KEY (dep_ID),
    UNIQUE KEY uq_dep_name (dep_Name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='科室信息';

CREATE TABLE IF NOT EXISTS doctor (
    d_ID          INT NOT NULL AUTO_INCREMENT,
    d_Name        VARCHAR(50) NOT NULL,
    d_Sex         CHAR(1) DEFAULT '男',
    d_Profession  VARCHAR(50)  COMMENT '职称',
    d_LoginName   VARCHAR(50),
    d_LoginPSW    VARCHAR(50),
    dep_ID        INT NOT NULL,
    PRIMARY KEY (d_ID),
    KEY idx_doctor_dept (dep_ID),
    CONSTRAINT fk_doctor_dept FOREIGN KEY (dep_ID) REFERENCES department(dep_ID)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='医生信息';

CREATE TABLE IF NOT EXISTS patientstatus (
    ps_ID     INT NOT NULL AUTO_INCREMENT,
    ps_Name   VARCHAR(20) NOT NULL,
    ps_Remark VARCHAR(100),
    PRIMARY KEY (ps_ID)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='就诊状态';

CREATE TABLE IF NOT EXISTS patient (
    p_ID       INT NOT NULL AUTO_INCREMENT,
    p_Name     VARCHAR(50) NOT NULL,
    p_Sex      CHAR(1) DEFAULT '男',
    p_Address  VARCHAR(50),
    p_Birth    DATETIME,
    ps_ID      INT,
    PRIMARY KEY (p_ID),
    CONSTRAINT fk_patient_status FOREIGN KEY (ps_ID) REFERENCES patientstatus(ps_ID)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='病人信息';

CREATE TABLE IF NOT EXISTS diagnosis (
    dia_ID         INT NOT NULL AUTO_INCREMENT,
    d_ID           INT,
    p_ID           INT,
    dia_Time       DATETIME,
    dia_Symptom    TEXT,
    dia_Diagnosis  TEXT,
    dia_Dispense   TEXT,
    dia_Remark     TEXT,
    PRIMARY KEY (dia_ID),
    CONSTRAINT fk_diag_doctor  FOREIGN KEY (d_ID) REFERENCES doctor(d_ID),
    CONSTRAINT fk_diag_patient FOREIGN KEY (p_ID) REFERENCES patient(p_ID)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='诊疗信息';

CREATE TABLE IF NOT EXISTS worker (
    w_ID         INT NOT NULL AUTO_INCREMENT,
    w_Name       VARCHAR(20),
    w_LoginName  VARCHAR(50),
    w_LoginPSW   VARCHAR(50),
    PRIMARY KEY (w_ID)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='挂号员';

-- 排班表（号源核心）—— 须在 register 之前建
CREATE TABLE IF NOT EXISTS schedule (
    sch_id         INT NOT NULL AUTO_INCREMENT,
    d_id           INT NOT NULL,
    sch_date       DATE NOT NULL,
    sch_time_slot  VARCHAR(20) NOT NULL  COMMENT '09:00-11:00|14:00-17:00|19:00-21:00',
    sch_type       VARCHAR(20) NOT NULL  COMMENT '专家|普通',
    sch_available  INT NOT NULL,
    sch_total      INT NOT NULL,
    PRIMARY KEY (sch_id),
    UNIQUE KEY uq_schedule (d_id, sch_date, sch_time_slot, sch_type),
    KEY idx_schedule_lookup (sch_date, sch_type),
    CONSTRAINT chk_available CHECK (sch_available >= 0),
    CONSTRAINT fk_schedule_doctor FOREIGN KEY (d_id) REFERENCES doctor(d_ID)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='医生排班/号源';

CREATE TABLE IF NOT EXISTS register (
    reg_ID      INT NOT NULL AUTO_INCREMENT,
    dep_ID      INT NOT NULL,
    p_ID        INT NOT NULL,
    w_ID        INT,
    d_ID        INT NOT NULL,
    sch_id      INT NOT NULL,
    reg_Time    DATETIME NOT NULL,
    reg_Fee     INT NOT NULL,
    reg_Order   INT,
    reg_Status  TINYINT DEFAULT 1  COMMENT '1=正常 0=已取消',
    PRIMARY KEY (reg_ID),
    KEY idx_register_patient (p_ID, reg_Time, reg_Status),
    CONSTRAINT fk_reg_dept    FOREIGN KEY (dep_ID) REFERENCES department(dep_ID),
    CONSTRAINT fk_reg_patient FOREIGN KEY (p_ID)   REFERENCES patient(p_ID),
    CONSTRAINT fk_reg_doctor  FOREIGN KEY (d_ID)   REFERENCES doctor(d_ID),
    CONSTRAINT fk_reg_sched   FOREIGN KEY (sch_id) REFERENCES schedule(sch_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='挂号记录';

-- ========== 补充表 ==========

CREATE TABLE IF NOT EXISTS user_account (
    ua_id         INT NOT NULL AUTO_INCREMENT,
    ua_Name       VARCHAR(50) NOT NULL,
    ua_Phone      VARCHAR(20),
    ua_Email      VARCHAR(100),
    created_time  DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ua_id),
    UNIQUE KEY uq_phone (ua_Phone)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户账户';

CREATE TABLE IF NOT EXISTS family_relation (
    fr_id          INT NOT NULL AUTO_INCREMENT,
    user_id        INT NOT NULL,
    patient_id     INT NOT NULL,
    relation_type  VARCHAR(20) NOT NULL  COMMENT '自己|父亲|母亲|配偶|儿子|女儿',
    alias          VARCHAR(50)           COMMENT '昵称：大宝、二宝',
    PRIMARY KEY (fr_id),
    UNIQUE KEY uq_user_alias (user_id, alias),
    CONSTRAINT fk_fr_user    FOREIGN KEY (user_id)    REFERENCES user_account(ua_id),
    CONSTRAINT fk_fr_patient FOREIGN KEY (patient_id) REFERENCES patient(p_ID)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='家属关系';

SET FOREIGN_KEY_CHECKS = 1;
