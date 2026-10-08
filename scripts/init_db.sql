-- ============================================================================
-- 教师智能问答系统 — MySQL 建表脚本
-- 库：subjects_kg（对应 config.ini [mysql].database）
--
-- 严格对应《教师智能问答系统技术文档》第 4.2 节的表定义：
--   subjects      学科配置表
--   chat_session  会话表
--   chat_record   问答记录表
--
-- 执行方式（需 root 或具备建库权限的账号）：
--     mysql -u root -p < scripts/init_db.sql
-- ============================================================================

CREATE DATABASE IF NOT EXISTS subjects_kg
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_0900_ai_ci;

USE subjects_kg;

-- ---------------------------------------------------------------------------
-- 表1：学科配置表 subjects（技术文档 4.2.1）
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `subjects` (
  `id`            INT UNSIGNED NOT NULL AUTO_INCREMENT,
  `subject_code`  VARCHAR(20)  NOT NULL COMMENT '学科代码，须在 valid_sources 白名单内',
  `subject_name`  VARCHAR(50)  NOT NULL COMMENT '学科显示名称',
  `description`   VARCHAR(255) DEFAULT NULL COMMENT '学科描述/覆盖知识点',
  `is_active`     TINYINT(1)   NOT NULL DEFAULT 1 COMMENT '是否启用',
  `sort_order`    INT          NOT NULL DEFAULT 0 COMMENT '展示排序',
  `created_at`    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_subject_code` (`subject_code`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='学科配置表';

-- 用 INSERT ... ON DUPLICATE KEY UPDATE 保证脚本可重复执行
INSERT INTO `subjects` (`subject_code`, `subject_name`, `description`, `sort_order`) VALUES
('ai',      '人工智能', '机器学习、深度学习、NLP、大模型应用',   1),
('java',    'Java开发', 'Java 基础、Spring 全家桶、微服务',      2),
('test',    '软件测试', '功能测试、自动化测试、性能测试',        3),
('ops',     '运维开发', 'Linux、Docker、Kubernetes、CI/CD',      4),
('bigdata', '大数据',   'Hadoop、Spark、Flink、数据仓库',        5)
ON DUPLICATE KEY UPDATE
  `subject_name` = VALUES(`subject_name`),
  `description`  = VALUES(`description`),
  `sort_order`   = VALUES(`sort_order`);

-- ---------------------------------------------------------------------------
-- 表2：会话表 chat_session（技术文档 4.2.2）
-- 注意：不存储用户真实身份信息（PRD 4.4.1），仅记录 IP 与 UA
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `chat_session` (
  `session_id`     CHAR(36)     NOT NULL COMMENT 'UUIDv4，服务端生成的会话标识',
  `subject_code`   VARCHAR(20)  DEFAULT NULL COMMENT '当前学科，NULL 表示全学科',
  `title`          VARCHAR(100) DEFAULT NULL COMMENT '会话标题（取首个问题前 30 字）',
  `message_count`  INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '问答轮次',
  `client_ip`      VARCHAR(45)  DEFAULT NULL COMMENT '客户端 IP（IPv6 兼容）',
  `user_agent`     VARCHAR(255) DEFAULT NULL COMMENT '客户端 UA',
  `is_deleted`     TINYINT(1)   NOT NULL DEFAULT 0 COMMENT '软删除标记',
  `created_at`     TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `last_active_at` TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
                                ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`session_id`),
  KEY `idx_last_active` (`last_active_at`),
  KEY `idx_subject` (`subject_code`, `last_active_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='会话表';

-- ---------------------------------------------------------------------------
-- 表3：问答记录表 chat_record（技术文档 4.2.3）
-- 字段设计直接支撑 PRD 第 6 章的统计需求（token 用量、缓存命中、溯源命中率等）
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `chat_record` (
  `record_id`         BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `session_id`        CHAR(36)     NOT NULL,
  `subject_code`      VARCHAR(20)  DEFAULT NULL COMMENT '本次问答使用的学科',
  `user_question`     VARCHAR(2000) NOT NULL COMMENT '用户问题（≤2000 字符）',
  `answer`            MEDIUMTEXT   COMMENT '助手答案',
  `references_json`   JSON         DEFAULT NULL COMMENT '溯源信息（父块摘要）',
  `retrieved_count`   INT          NOT NULL DEFAULT 0 COMMENT '召回子块数',
  `is_greeting`       TINYINT(1)   NOT NULL DEFAULT 0 COMMENT '是否问候语分流',
  `is_from_cache`     TINYINT(1)   NOT NULL DEFAULT 0 COMMENT '是否命中缓存',
  `model_name`        VARCHAR(50)  DEFAULT NULL COMMENT '实际使用的模型',
  `prompt_tokens`     INT          DEFAULT 0,
  `completion_tokens` INT          DEFAULT 0,
  `response_time_ms`  INT          NOT NULL DEFAULT 0 COMMENT '端到端耗时（毫秒）',
  `finish_reason`     VARCHAR(20)  DEFAULT 'stop' COMMENT 'stop/interrupted/error',
  `error_code`        VARCHAR(40)  DEFAULT NULL,
  `created_at`        TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`record_id`),
  KEY `idx_session_time` (`session_id`, `created_at` DESC),
  KEY `idx_created` (`created_at`),
  KEY `idx_subject_created` (`subject_code`, `created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='问答记录表';

-- ---------------------------------------------------------------------------
-- 授权：应用账号 edu_rag（对应 config.ini [mysql].user）
-- 最小权限：仅授予本库 CRUD，不给 DROP / GRANT 等管理权限
--
-- ⚠️ 密码不要硬编码在本文件里（会随仓库公开）。两种做法二选一：
--   1) 执行前先设置变量（推荐）：
--        mysql -u root -p < scripts/init_db.sql
--        或在客户端中：
--        SET @app_password = '你的强密码';
--   2) 直接手写一个强密码替换下面的占位符，并同步写入 .env 的
--      EDU_MYSQL_PASSWORD。切勿把真实密码提交进仓库。
-- ---------------------------------------------------------------------------
-- 若未设置 @app_password，用一个明显的占位值，强制使用者修改
SET @app_password = IFNULL(@app_password, 'CHANGE_ME_BEFORE_USE');

-- 用预处理语句拼 SQL：MySQL 不允许在 CREATE USER 里直接用变量
SET @sql_create_user = CONCAT(
  "CREATE USER IF NOT EXISTS 'edu_rag'@'%' IDENTIFIED BY '", @app_password, "'"
);
PREPARE stmt FROM @sql_create_user; EXECUTE stmt; DEALLOCATE PREPARE stmt;

SET @sql_create_user_local = CONCAT(
  "CREATE USER IF NOT EXISTS 'edu_rag'@'localhost' IDENTIFIED BY '", @app_password, "'"
);
PREPARE stmt FROM @sql_create_user_local; EXECUTE stmt; DEALLOCATE PREPARE stmt;

GRANT SELECT, INSERT, UPDATE, DELETE ON subjects_kg.* TO 'edu_rag'@'%';
GRANT SELECT, INSERT, UPDATE, DELETE ON subjects_kg.* TO 'edu_rag'@'localhost';
FLUSH PRIVILEGES;

SELECT '建表完成' AS status;
SHOW TABLES;
SELECT subject_code, subject_name FROM subjects ORDER BY sort_order;
