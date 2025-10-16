// MCP Classification Study - With CLTools Validation Questions
// Generated dynamically from CLServers CSV and CLTools hierarchy
// DO NOT EDIT MANUALLY - Regenerate using generate_main_ts.py

// import gorilla = require("gorilla/gorilla");  // Commented out for browser compatibility

// O*NET Hierarchy Data Structures
var onetData = {
    l1Clusters: {
        'L1_01': 'Business management, finance, and customer service operations',
        'L1_02': 'Comprehensive healthcare services and medical specialties',
        'L1_03': 'Manage education, HR, and professional development programs',
        'L1_04': 'Design, implement, and maintain diverse information technology systems',
        'L1_05': 'Operate and manage diverse industrial and agricultural processes',
        'L1_06': 'Perform government regulatory enforcement and public safety operations',
        'L1_07': 'Conduct scientific research and technical analysis across disciplines',
        'L1_08': 'Create and preserve art, culture, and religious artifacts',
        'L1_09': 'Coordinate transportation networks and manage logistics supply chains',
        'L1_10': 'Manage diverse energy sources and optimize power systems',
        'L1_11': 'Design and construct infrastructure projects and engineering systems',
        'L1_12': 'Manage and improve environmental systems and sustainability practices',
    },

    l2ToL1: {
        '0.0': 'L1_11',
        '10.0': 'L1_07',
        '100.0': 'L1_04',
        '103.0': 'L1_11',
        '105.0': 'L1_03',
        '107.0': 'L1_08',
        '108.0': 'L1_03',
        '109.0': 'L1_08',
        '113.0': 'L1_02',
        '114.0': 'L1_07',
        '115.0': 'L1_11',
        '116.0': 'L1_03',
        '118.0': 'L1_08',
        '119.0': 'L1_07',
        '120.0': 'L1_07',
        '122.0': 'L1_05',
        '126.0': 'L1_03',
        '128.0': 'L1_04',
        '13.0': 'L1_11',
        '131.0': 'L1_03',
        '132.0': 'L1_11',
        '135.0': 'L1_06',
        '136.0': 'L1_04',
        '137.0': 'L1_03',
        '139.0': 'L1_11',
        '140.0': 'L1_11',
        '142.0': 'L1_03',
        '143.0': 'L1_10',
        '144.0': 'L1_06',
        '146.0': 'L1_05',
        '147.0': 'L1_09',
        '149.0': 'L1_06',
        '152.0': 'L1_05',
        '153.0': 'L1_06',
        '154.0': 'L1_07',
        '155.0': 'L1_01',
        '156.0': 'L1_06',
        '157.0': 'L1_11',
        '159.0': 'L1_11',
        '160.0': 'L1_07',
        '161.0': 'L1_01',
        '162.0': 'L1_05',
        '163.0': 'L1_07',
        '164.0': 'L1_06',
        '167.0': 'L1_03',
        '168.0': 'L1_07',
        '169.0': 'L1_07',
        '17.0': 'L1_09',
        '174.0': 'L1_03',
        '175.0': 'L1_06',
        '178.0': 'L1_08',
        '18.0': 'L1_07',
        '180.0': 'L1_05',
        '183.0': 'L1_12',
        '186.0': 'L1_03',
        '187.0': 'L1_08',
        '189.0': 'L1_04',
        '19.0': 'L1_07',
        '191.0': 'L1_02',
        '193.0': 'L1_04',
        '195.0': 'L1_01',
        '196.0': 'L1_07',
        '197.0': 'L1_03',
        '198.0': 'L1_04',
        '2.0': 'L1_11',
        '20.0': 'L1_02',
        '202.0': 'L1_06',
        '205.0': 'L1_03',
        '21.0': 'L1_01',
        '211.0': 'L1_03',
        '213.0': 'L1_05',
        '214.0': 'L1_05',
        '215.0': 'L1_02',
        '218.0': 'L1_07',
        '220.0': 'L1_06',
        '223.0': 'L1_08',
        '225.0': 'L1_05',
        '229.0': 'L1_03',
        '23.0': 'L1_02',
        '230.0': 'L1_03',
        '232.0': 'L1_03',
        '233.0': 'L1_01',
        '234.0': 'L1_05',
        '236.0': 'L1_07',
        '237.0': 'L1_07',
        '238.0': 'L1_06',
        '24.0': 'L1_05',
        '241.0': 'L1_09',
        '243.0': 'L1_01',
        '246.0': 'L1_11',
        '249.0': 'L1_07',
        '25.0': 'L1_04',
        '250.0': 'L1_11',
        '252.0': 'L1_03',
        '253.0': 'L1_06',
        '254.0': 'L1_11',
        '257.0': 'L1_04',
        '258.0': 'L1_03',
        '259.0': 'L1_07',
        '26.0': 'L1_06',
        '260.0': 'L1_04',
        '262.0': 'L1_07',
        '263.0': 'L1_12',
        '264.0': 'L1_11',
        '265.0': 'L1_02',
        '267.0': 'L1_11',
        '269.0': 'L1_09',
        '27.0': 'L1_05',
        '272.0': 'L1_03',
        '274.0': 'L1_02',
        '277.0': 'L1_07',
        '278.0': 'L1_07',
        '28.0': 'L1_03',
        '283.0': 'L1_03',
        '285.0': 'L1_03',
        '287.0': 'L1_08',
        '288.0': 'L1_04',
        '29.0': 'L1_01',
        '291.0': 'L1_11',
        '293.0': 'L1_04',
        '295.0': 'L1_05',
        '297.0': 'L1_09',
        '299.0': 'L1_11',
        '301.0': 'L1_03',
        '302.0': 'L1_03',
        '308.0': 'L1_09',
        '31.0': 'L1_07',
        '310.0': 'L1_04',
        '312.0': 'L1_02',
        '313.0': 'L1_05',
        '314.0': 'L1_08',
        '317.0': 'L1_04',
        '318.0': 'L1_02',
        '319.0': 'L1_12',
        '320.0': 'L1_08',
        '323.0': 'L1_07',
        '326.0': 'L1_03',
        '327.0': 'L1_04',
        '329.0': 'L1_06',
        '333.0': 'L1_06',
        '336.0': 'L1_03',
        '337.0': 'L1_03',
        '338.0': 'L1_02',
        '339.0': 'L1_07',
        '34.0': 'L1_05',
        '340.0': 'L1_12',
        '341.0': 'L1_03',
        '342.0': 'L1_04',
        '343.0': 'L1_02',
        '348.0': 'L1_11',
        '349.0': 'L1_05',
        '35.0': 'L1_02',
        '350.0': 'L1_09',
        '352.0': 'L1_06',
        '353.0': 'L1_03',
        '354.0': 'L1_07',
        '355.0': 'L1_07',
        '356.0': 'L1_09',
        '357.0': 'L1_03',
        '360.0': 'L1_05',
        '364.0': 'L1_01',
        '366.0': 'L1_02',
        '368.0': 'L1_01',
        '369.0': 'L1_08',
        '37.0': 'L1_01',
        '370.0': 'L1_01',
        '371.0': 'L1_02',
        '372.0': 'L1_06',
        '375.0': 'L1_03',
        '378.0': 'L1_06',
        '379.0': 'L1_02',
        '381.0': 'L1_10',
        '383.0': 'L1_07',
        '384.0': 'L1_07',
        '387.0': 'L1_01',
        '393.0': 'L1_07',
        '394.0': 'L1_08',
        '398.0': 'L1_04',
        '40.0': 'L1_11',
        '41.0': 'L1_03',
        '44.0': 'L1_07',
        '46.0': 'L1_06',
        '48.0': 'L1_09',
        '52.0': 'L1_02',
        '58.0': 'L1_02',
        '60.0': 'L1_02',
        '61.0': 'L1_01',
        '63.0': 'L1_01',
        '66.0': 'L1_07',
        '68.0': 'L1_05',
        '7.0': 'L1_12',
        '70.0': 'L1_04',
        '71.0': 'L1_04',
        '74.0': 'L1_06',
        '76.0': 'L1_02',
        '77.0': 'L1_04',
        '8.0': 'L1_02',
        '81.0': 'L1_03',
        '83.0': 'L1_09',
        '86.0': 'L1_03',
        '88.0': 'L1_12',
        '89.0': 'L1_10',
        '9.0': 'L1_03',
        '91.0': 'L1_05',
        '92.0': 'L1_01',
        '95.0': 'L1_08',
        '96.0': 'L1_04',
        '97.0': 'L1_07',
    },

    l2Clusters: {
        '0.0': '\'Automotive Systems Engineering Design Development and Testing\'',
        '10.0': '\'Market Research and Consumer Data Analysis\'',
        '100.0': '\'Clinical Data Management and Regulatory Compliance Operations\'',
        '103.0': '\'Project Cost Estimation and Budget Analysis\'',
        '105.0': '\'Professional Development and Continuing Education Activities\'',
        '107.0': '\'Musical Direction and Composition for Performance Groups\'',
        '108.0': '\'Student Record Keeping and Attendance Management\'',
        '109.0': '\'Tour Guide and Group Leadership Services\'',
        '113.0': '\'Cardiovascular Diagnosis Treatment and Emergency Cardiac Care\'',
        '114.0': '\'Bioinformatics Database Development and Computational Analysis\'',
        '115.0': '\'Masonry and Concrete Construction and Repair\'',
        '116.0': '\'Academic and Career Counseling and Student Organization Advising\'',
        '118.0': '\'Social and Cultural Research Through Observation and Data Collection\'',
        '119.0': '\'Medical Imaging and Radiographic Diagnostic Procedures\'',
        '120.0': '\'Geological Survey and Resource Exploration Analysis\'',
        '122.0': '\'Measuring, Mixing, and Baking Food Products and Ingredients\'',
        '126.0': '\'Organizational Leadership and Operations Management\'',
        '128.0': '\'Database Design, Development, and Administration Management\'',
        '13.0': '\'Water Resources Engineering and Management\'',
        '131.0': '\'Lecture Preparation and Course Material Development\'',
        '132.0': '\'Engineering Design and Development of Mechanical and Automated Systems\'',
        '135.0': '\'Tax Assessment, Audit, and Compliance Services\'',
        '136.0': '\'Creative Project Management and Design Coordination\'',
        '137.0': '\'Student Assignment and Coursework Evaluation and Grading\'',
        '139.0': '\'Robotic Systems Design, Development, and Technical Support\'',
        '140.0': '\'Architectural Design and Construction Planning Services\'',
        '142.0': '\'Fitness and Wellness Program Management and Coordination\'',
        '143.0': '\'Solar Energy System Design, Installation, and Technical Support\'',
        '144.0': '\'Legal Dispute Resolution and Claims Adjudication\'',
        '146.0': '\'Extruding, Molding, and Forming Machine Operation and Setup\'',
        '147.0': '\'Travel Planning and Reservation Services\'',
        '149.0': '\'Parking Enforcement and Vehicle Management Services\'',
        '152.0': '\'Heavy Equipment Operation and Mining Machinery Control\'',
        '153.0': '\'Legal Representation and Court Proceedings Management\'',
        '154.0': '\'Research Data Management and Statistical Analysis Support\'',
        '155.0': '\'Customer Service and Complaint Resolution\'',
        '156.0': '\'Government Property and Title Investigation and Compliance\'',
        '157.0': '\'Structural Metal Fabrication, Assembly, and Installation\'',
        '159.0': '\'Electromechanical Systems Technical Support and Manufacturing Operations\'',
        '160.0': '\'Neuropsychological Assessment, Diagnosis, and Treatment of Brain-Related Conditions\'',
        '161.0': '\'Comprehensive Financial Planning and Debt Management Advisory Services\'',
        '162.0': '\'Animal Research, Breeding, and Production Management\'',
        '163.0': '\'News Reporting and Editorial Content Creation\'',
        '164.0': '\'Forest Fire Suppression and Prevention Management\'',
        '167.0': '\'Supervise Staff Operations and Facility Management Activities\'',
        '168.0': '\'Aerial Photography Interpretation and Cartographic Compilation\'',
        '169.0': '\'Academic Collaboration on Teaching and Research Issues\'',
        '17.0': '\'Transportation Coordination and Freight Management Services\'',
        '174.0': '\'Academic Department Head Administrative Duties and Teaching Responsibilities\'',
        '175.0': '\'Offender Supervision and Correctional Case Management\'',
        '178.0': '\'Professional Writing and Content Development Services\'',
        '18.0': '\'Compiling Specialized Bibliographies for Student Reading Assignments\'',
        '180.0': '\'Production and Operational Data Recording\'',
        '183.0': '\'Forest Management and Conservation Planning\'',
        '186.0': '\'Comprehensive Childcare and Development Services\'',
        '187.0': '\'Professional Photography and Image Processing Services\'',
        '189.0': '\'Computer Systems Analysis, Design, and Programming Management\'',
        '19.0': '\'Medical Equipment Maintenance, Calibration, and Training Management\'',
        '191.0': '\'Surgical Procedures and Operative Patient Care\'',
        '193.0': '\'Healthcare Information Systems Development and Management\'',
        '195.0': '\'Cash Handling and Payment Processing Operations\'',
        '196.0': '\'Research Communication and Dissemination\'',
        '197.0': '\'Budget Planning, Development, and Financial Control Management\'',
        '198.0': '\'Geographic Information Systems Development and Analysis\'',
        '2.0': '\'Blueprint and Specification Interpretation for Project Planning\'',
        '20.0': '\'Medical Equipment and Supply Management Operations\'',
        '202.0': '\'Application Processing and Eligibility Determination Through Interviews and Documentation Review\'',
        '205.0': '\'Speech-Language Pathology Assessment, Treatment, and Documentation Services\'',
        '21.0': '\'Telephone and Communication System Operations\'',
        '211.0': '\'Employee Benefits and Compensation Administration and Policy Development\'',
        '213.0': '\'Production and Operations Supervision with Workforce Management\'',
        '214.0': '\'Grounds and Landscaping Maintenance Supervision and Operations\'',
        '215.0': '\'Comprehensive Maternal and Newborn Care Services\'',
        '218.0': '\'Fraud Investigation and Financial Crime Detection\'',
        '220.0': '\'Criminal Investigation and Evidence Collection\'',
        '223.0': '\'Religious Ministry and Spiritual Leadership Services\'',
        '225.0': '\'Valve and Flow Control Operations\'',
        '229.0': '\'Event Planning and Management Services\'',
        '23.0': '\'Physical Therapy Assessment, Treatment Planning, and Therapeutic Intervention Services\'',
        '230.0': '\'Organize and Lead Recreational Activities for Physical Mental Social Development\'',
        '232.0': '\'Teaching and Instructional Support for Students with Diverse Learning Needs\'',
        '233.0': '\'Securities Trading and Investment Advisory Services\'',
        '234.0': '\'Food Preparation and Cooking Operations\'',
        '236.0': '\'Business Operations Analysis and Process Improvement\'',
        '237.0': '\'Professional Translation and Interpretation Services\'',
        '238.0': '\'Security Screening and Access Control Operations\'',
        '24.0': '\'Machinery and Equipment Maintenance, Repair, and Troubleshooting Operations\'',
        '241.0': '\'Packaging and Container Preparation for Shipping and Storage\'',
        '243.0': '\'Hotel Operations Management and Guest Services\'',
        '246.0': '\'Security System Installation and Maintenance Services\'',
        '249.0': '\'Biomedical Research and Development in Life Sciences\'',
        '25.0': '\'System Architecture Design and Hardware-Software Integration\'',
        '250.0': '\'Technical Drawing and CAD Design Development\'',
        '252.0': '\'Academic and Administrative Committee Service\'',
        '253.0': '\'Wildlife Conservation and Park Visitor Services Management\'',
        '254.0': '\'HVAC System Installation Testing and Energy Efficiency Optimization\'',
        '257.0': '\'Cybersecurity Defense and Digital Forensics Operations\'',
        '258.0': '\'Recreation and Entertainment Facility Management and Operations\'',
        '259.0': '\'Laboratory Specimen Collection and Analysis\'',
        '26.0': '\'Regulatory Compliance Management and Agency Relations\'',
        '260.0': '\'Network Infrastructure Design and Management\'',
        '262.0': '\'Conduct Research and Publish Findings in Professional Publications\'',
        '263.0': '\'Sustainability Program Development and Implementation Management\'',
        '264.0': '\'Carpentry and Woodworking Construction and Assembly\'',
        '265.0': '\'Patient Referral and Professional Consultation Services\'',
        '267.0': '\'Electrical and Electronics Engineering Design and Implementation\'',
        '269.0': '\'Railroad Operations and Security Management\'',
        '27.0': '\'Computer Numerical Control Programming and Machine Operation\'',
        '272.0': '\'Educational Program Administration and Strategic Management\'',
        '274.0': '\'Medical Records Documentation and Management\'',
        '277.0': '\'Histological Specimen Preparation and Microscopic Tissue Analysis\'',
        '278.0': '\'Financial Analysis and Quantitative Modeling for Risk Assessment\'',
        '28.0': '\'Student Assessment, Consultation, and Behavioral Support Services\'',
        '283.0': '\'Examination Development, Administration, and Grading Management\'',
        '285.0': '\'Payroll Processing and Employee Compensation Management\'',
        '287.0': '\'Artistic Creation and Visual Design Development\'',
        '288.0': '\'Game Design and Development Management\'',
        '29.0': '\'Cash Handling and Banking Transaction Processing\'',
        '291.0': '\'Transportation Planning and Infrastructure Design\'',
        '293.0': '\'Network Infrastructure Management and Technical Support\'',
        '295.0': '\'Agricultural and Commercial Procurement and Supply Chain Management\'',
        '297.0': '\'Vehicle Maintenance, Inspection, and Operational Compliance Management\'',
        '299.0': '\'Product Design Development and Market Strategy\'',
        '301.0': '\'Strategic Human Resources Management and Organizational Development\'',
        '302.0': '\'Student Recruitment, Registration, and Enrollment Activities\'',
        '308.0': '\'Logistics Operations Management and Customer Relationship Coordination\'',
        '31.0': '\'Research Grant Proposal Writing and Funding Procurement\'',
        '310.0': '\'Business Continuity and Disaster Recovery Planning and Implementation\'',
        '312.0': '\'Comprehensive Eye Care Diagnosis Treatment and Vision Correction Services\'',
        '313.0': '\'Chemical Process Equipment Operation and Monitoring\'',
        '314.0': '\'Theatrical and Film Costume Design, Construction, and Management\'',
        '317.0': '\'Project Planning, Coordination, and Management\'',
        '318.0': '\'Sports Medicine and Athletic Healthcare Management\'',
        '319.0': '\'Environmental Compliance Monitoring and Enforcement Operations\'',
        '320.0': '\'Musical Instrument Repair and Maintenance Services\'',
        '323.0': '\'Atmospheric and Astronomical Research and Analysis\'',
        '326.0': '\'Athletic Training and Exercise Program Development\'',
        '327.0': '\'Data Warehouse and Document Management Systems Development\'',
        '329.0': '\'Emergency and Service Dispatch Coordination and Communication\'',
        '333.0': '\'Government Relations and Legislative Affairs Management\'',
        '336.0': '\'Academic Leadership and Professional Service Activities\'',
        '337.0': '\'Facilitating and Moderating Classroom Discussions\'',
        '338.0': '\'Personal Care and Daily Living Assistance for Patients\'',
        '339.0': '\'Clinical Research Coordination and Protocol Management\'',
        '34.0': '\'Food Processing Equipment Operation and Monitoring\'',
        '340.0': '\'Waste Collection, Sorting, and Recycling Operations Management\'',
        '341.0': '\'Academic Record Keeping and Administrative Reporting\'',
        '342.0': '\'Web Site Development and Administration\'',
        '343.0': '\'Primary Care and Preventive Medicine Practice\'',
        '348.0': '\'Product Development and Manufacturing Process Engineering\'',
        '349.0': '\'Industrial Material Processing and Equipment Operation\'',
        '35.0': '\'Comprehensive Nursing Care Delivery and Clinical Practice Management\'',
        '350.0': '\'Inventory Management and Stock Replenishment Operations\'',
        '352.0': '\'Law Enforcement Patrol and Public Safety Operations\'',
        '353.0': '\'Curriculum Planning and Learning Objective Development\'',
        '354.0': '\'Food Science Research and Quality Assurance Testing\'',
        '355.0': '\'Editorial Review and Publication Preparation\'',
        '356.0': '\'Air Traffic Control and Flight Coordination Operations\'',
        '357.0': '\'Student Assessment, Instruction, and Academic Support Services\'',
        '360.0': '\'Agricultural Engineering and Precision Farming Technology Management\'',
        '364.0': '\'Bartending and Beverage Service Operations\'',
        '366.0': '\'Dermatological and Allergic Immunologic Diagnosis and Treatment\'',
        '368.0': '\'Online Retail Operations and E-commerce Management\'',
        '369.0': '\'Museum and Archive Collection Management and Preservation\'',
        '37.0': '\'Loan Processing and Credit Evaluation Services\'',
        '370.0': '\'Credit Assessment and Debt Collection Management\'',
        '371.0': '\'Anesthesia Administration and Patient Monitoring During Medical Procedures\'',
        '372.0': '\'Security Operations Management and Risk Assessment\'',
        '375.0': '\'Educational Technology Integration and Multimedia Instruction\'',
        '378.0': '\'Emergency Management Planning and Disaster Response Coordination\'',
        '379.0': '\'Medical Administrative Support and Patient Services\'',
        '381.0': '\'Energy Auditing and Conservation Assessment Services\'',
        '383.0': '\'Crime Scene Evidence Collection and Forensic Analysis\'',
        '384.0': '\'Genetic Research and Laboratory Analysis\'',
        '387.0': '\'Customer Service Records and Transaction Documentation\'',
        '393.0': '\'Software Testing and Quality Assurance Analysis\'',
        '394.0': '\'Music Therapy Treatment Design and Implementation\'',
        '398.0': '\'Library Operations and Patron Services Management\'',
        '40.0': '\'Construction Project Management and Coordination\'',
        '41.0': '\'Campus and Community Engagement Activities\'',
        '44.0': '\'Statistical Analysis and Mathematical Modeling for Research and Business Applications\'',
        '46.0': '\'Fire Emergency Response and Firefighting Operations Management\'',
        '48.0': '\'Logistics Data Analysis and Performance Optimization\'',
        '52.0': '\'Mental Health Assessment, Treatment, and Patient Care Services\'',
        '58.0': '\'Manual Therapy and Alternative Medicine Treatment Services\'',
        '60.0': '\'Public Health Program Development and Community Health Education\'',
        '61.0': '\'Restaurant Food and Beverage Service Operations\'',
        '63.0': '\'Customer Relationship Management and Sales Contract Development\'',
        '66.0': '\'Chemical Process Development and Manufacturing Operations\'',
        '68.0': '\'Agricultural Operations Management and Production Oversight\'',
        '7.0': '\'Research and Development Management for Natural Resources and Environmental Sciences\'',
        '70.0': '\'Audio-Visual Equipment Operation, Setup, and Technical Production Management\'',
        '71.0': '\'IT Operations Management and Technical Support\'',
        '74.0': '\'Administrative Processing and Public Service for Legal and Municipal Operations\'',
        '76.0': '\'Medical Education and Training of Healthcare Professionals\'',
        '77.0': '\'Cybersecurity Assessment, Testing, and Implementation Management\'',
        '8.0': '\'Medical Diagnosis Through Examination Testing and Diagnostic Interpretation\'',
        '81.0': '\'Maintain Regular Office Hours for Student Advising and Assistance\'',
        '83.0': '\'Professional Vehicle Transportation and Passenger Services\'',
        '86.0': '\'Educational Materials Procurement and Inventory Management\'',
        '88.0': '\'Environmental Impact Assessment and Industrial Ecology Analysis\'',
        '89.0': '\'Power System Operations and Distribution Control\'',
        '9.0': '\'Employee Training Program Development and Implementation\'',
        '91.0': '\'Quality Control Management and Production Standards Oversight\'',
        '92.0': '\'Financial Management and Accounting Operations Oversight\'',
        '95.0': '\'Surface Cleaning and Decontamination Using Chemical and Mechanical Methods\'',
        '96.0': '\'Web Development and E-commerce Platform Design\'',
        '97.0': '\'Remote Sensing Data Analysis and Geospatial Information Management\'',
    },

    taskToL2: {
        '1.0': '155.0',
        '10.0': '63.0',
        '1000.0': '301.0',
        '1002.0': '301.0',
        '1003.0': '301.0',
        '10039.0': '157.0',
        '10059.0': '34.0',
        '1014.0': '9.0',
        '10160.0': '180.0',
        '10175.0': '146.0',
        '10177.0': '146.0',
        '10189.0': '146.0',
        '10219.0': '180.0',
        '10295.0': '157.0',
        '10494.0': '313.0',
        '10495.0': '313.0',
        '10496.0': '180.0',
        '10637.0': '297.0',
        '10644.0': '297.0',
        '10671.0': '269.0',
        '10672.0': '269.0',
        '10674.0': '269.0',
        '10678.0': '269.0',
        '10680.0': '269.0',
        '10691.0': '269.0',
        '10697.0': '269.0',
        '10698.0': '269.0',
        '10702.0': '269.0',
        '10704.0': '269.0',
        '10718.0': '155.0',
        '10723.0': '387.0',
        '10835.0': '319.0',
        '10838.0': '319.0',
        '10842.0': '319.0',
        '10902.0': '139.0',
        '10903.0': '44.0',
        '10904.0': '44.0',
        '10905.0': '44.0',
        '10906.0': '196.0',
        '10907.0': '44.0',
        '10908.0': '249.0',
        '10911.0': '323.0',
        '10934.0': '223.0',
        '10935.0': '223.0',
        '10937.0': '223.0',
        '10938.0': '223.0',
        '10940.0': '223.0',
        '10944.0': '223.0',
        '10951.0': '223.0',
        '10956.0': '137.0',
        '10958.0': '81.0',
        '10959.0': '137.0',
        '10960.0': '283.0',
        '10963.0': '131.0',
        '10967.0': '131.0',
        '10977.0': '68.0',
        '10989.0': '287.0',
        '10990.0': '287.0',
        '10991.0': '287.0',
        '10993.0': '287.0',
        '10994.0': '287.0',
        '10995.0': '287.0',
        '10996.0': '287.0',
        '10999.0': '287.0',
        '11000.0': '287.0',
        '11013.0': '287.0',
        '1104.0': '243.0',
        '11046.0': '178.0',
        '11047.0': '178.0',
        '11048.0': '178.0',
        '11049.0': '178.0',
        '1105.0': '243.0',
        '11051.0': '178.0',
        '11052.0': '178.0',
        '11055.0': '178.0',
        '11057.0': '107.0',
        '1106.0': '243.0',
        '11060.0': '178.0',
        '1107.0': '243.0',
        '1108.0': '243.0',
        '11102.0': '164.0',
        '11104.0': '164.0',
        '1111.0': '243.0',
        '1114.0': '243.0',
        '1115.0': '243.0',
        '11154.0': '61.0',
        '11156.0': '61.0',
        '11167.0': '61.0',
        '11172.0': '61.0',
        '11178.0': '61.0',
        '11189.0': '61.0',
        '1121.0': '243.0',
        '1122.0': '243.0',
        '11237.0': '63.0',
        '11238.0': '63.0',
        '11239.0': '63.0',
        '1124.0': '243.0',
        '11240.0': '63.0',
        '11241.0': '63.0',
        '11245.0': '63.0',
        '11246.0': '63.0',
        '11251.0': '63.0',
        '11252.0': '63.0',
        '11256.0': '63.0',
        '1126.0': '243.0',
        '11267.0': '63.0',
        '1129.0': '126.0',
        '11291.0': '37.0',
        '11292.0': '37.0',
        '11296.0': '37.0',
        '11300.0': '37.0',
        '11301.0': '37.0',
        '11304.0': '37.0',
        '11312.0': '381.0',
        '11315.0': '155.0',
        '11369.0': '103.0',
        '1137.0': '197.0',
        '11406.0': '387.0',
        '11409.0': '257.0',
        '11431.0': '180.0',
        '1151.0': '295.0',
        '11641.0': '27.0',
        '11710.0': '24.0',
        '11761.0': '246.0',
        '11821.0': '24.0',
        '11827.0': '27.0',
        '11865.0': '320.0',
        '11883.0': '320.0',
        '11890.0': '320.0',
        '11893.0': '320.0',
        '11896.0': '320.0',
        '11918.0': '122.0',
        '11937.0': '27.0',
        '11941.0': '27.0',
        '11944.0': '27.0',
        '11946.0': '27.0',
        '11947.0': '27.0',
        '11948.0': '27.0',
        '11961.0': '27.0',
        '12175.0': '314.0',
        '12215.0': '264.0',
        '12221.0': '27.0',
        '1224.0': '229.0',
        '1229.0': '229.0',
        '12301.0': '89.0',
        '12304.0': '89.0',
        '12306.0': '89.0',
        '12308.0': '89.0',
        '12309.0': '89.0',
        '12312.0': '89.0',
        '12336.0': '313.0',
        '12400.0': '349.0',
        '1252.0': '370.0',
        '1256.0': '370.0',
        '1257.0': '370.0',
        '1258.0': '370.0',
        '12584.0': '267.0',
        '1267.0': '189.0',
        '1268.0': '189.0',
        '1269.0': '189.0',
        '1270.0': '189.0',
        '1271.0': '189.0',
        '1272.0': '189.0',
        '12728.0': '356.0',
        '12729.0': '356.0',
        '1273.0': '189.0',
        '12730.0': '356.0',
        '12731.0': '356.0',
        '12733.0': '356.0',
        '12739.0': '356.0',
        '1274.0': '189.0',
        '12743.0': '356.0',
        '12746.0': '356.0',
        '1275.0': '189.0',
        '12755.0': '269.0',
        '1276.0': '189.0',
        '12760.0': '269.0',
        '12766.0': '269.0',
        '12767.0': '269.0',
        '1277.0': '189.0',
        '12770.0': '269.0',
        '12771.0': '269.0',
        '12773.0': '269.0',
        '1279.0': '189.0',
        '1280.0': '189.0',
        '1282.0': '293.0',
        '1283.0': '293.0',
        '1285.0': '71.0',
        '12878.0': '295.0',
        '12879.0': '295.0',
        '1288.0': '293.0',
        '1289.0': '71.0',
        '12891.0': '156.0',
        '12913.0': '161.0',
        '12929.0': '249.0',
        '12930.0': '249.0',
        '12932.0': '196.0',
        '12934.0': '249.0',
        '12935.0': '249.0',
        '12936.0': '249.0',
        '12937.0': '249.0',
        '12938.0': '249.0',
        '12940.0': '249.0',
        '12942.0': '249.0',
        '12943.0': '384.0',
        '12944.0': '249.0',
        '12946.0': '249.0',
        '12947.0': '249.0',
        '12951.0': '249.0',
        '12952.0': '154.0',
        '12953.0': '154.0',
        '12955.0': '154.0',
        '12956.0': '154.0',
        '12957.0': '154.0',
        '12958.0': '154.0',
        '12959.0': '154.0',
        '12960.0': '154.0',
        '12961.0': '154.0',
        '12963.0': '154.0',
        '12964.0': '154.0',
        '12965.0': '154.0',
        '12967.0': '154.0',
        '12968.0': '154.0',
        '12969.0': '154.0',
        '12970.0': '154.0',
        '12973.0': '154.0',
        '12983.0': '229.0',
        '12988.0': '223.0',
        '1299.0': '128.0',
        '12993.0': '144.0',
        '12995.0': '144.0',
        '12999.0': '144.0',
        '130.0': '140.0',
        '1300.0': '128.0',
        '13003.0': '144.0',
        '13005.0': '144.0',
        '1301.0': '128.0',
        '1302.0': '128.0',
        '1303.0': '128.0',
        '1305.0': '128.0',
        '1306.0': '128.0',
        '1309.0': '128.0',
        '13096.0': '220.0',
        '13097.0': '220.0',
        '131.0': '140.0',
        '1311.0': '128.0',
        '1312.0': '128.0',
        '1313.0': '128.0',
        '13146.0': '167.0',
        '1315.0': '128.0',
        '13151.0': '167.0',
        '13158.0': '109.0',
        '13160.0': '109.0',
        '1317.0': '71.0',
        '1318.0': '71.0',
        '1319.0': '71.0',
        '13191.0': '63.0',
        '13193.0': '63.0',
        '13198.0': '63.0',
        '1320.0': '71.0',
        '13200.0': '63.0',
        '13201.0': '63.0',
        '13203.0': '63.0',
        '13206.0': '63.0',
        '1321.0': '71.0',
        '1322.0': '71.0',
        '1323.0': '71.0',
        '13238.0': '63.0',
        '1324.0': '71.0',
        '13247.0': '21.0',
        '13248.0': '155.0',
        '1325.0': '71.0',
        '13257.0': '21.0',
        '13258.0': '21.0',
        '1327.0': '71.0',
        '1328.0': '71.0',
        '1329.0': '71.0',
        '1330.0': '71.0',
        '1333.0': '71.0',
        '1348.0': '267.0',
        '1349.0': '267.0',
        '136.0': '140.0',
        '13695.0': '120.0',
        '13741.0': '70.0',
        '13747.0': '70.0',
        '13837.0': '195.0',
        '13843.0': '387.0',
        '1409.0': '308.0',
        '1420.0': '132.0',
        '1447.0': '250.0',
        '1450.0': '250.0',
        '1451.0': '250.0',
        '1452.0': '250.0',
        '1455.0': '250.0',
        '1461.0': '140.0',
        '14613.0': '225.0',
        '14623.0': '189.0',
        '14624.0': '317.0',
        '14626.0': '189.0',
        '14627.0': '71.0',
        '14634.0': '71.0',
        '14635.0': '25.0',
        '14639.0': '393.0',
        '1464.0': '2.0',
        '14640.0': '393.0',
        '14641.0': '393.0',
        '14642.0': '393.0',
        '14644.0': '393.0',
        '14647.0': '393.0',
        '14649.0': '393.0',
        '14652.0': '393.0',
        '14654.0': '393.0',
        '14655.0': '393.0',
        '14656.0': '393.0',
        '14659.0': '393.0',
        '14661.0': '393.0',
        '14664.0': '393.0',
        '14669.0': '25.0',
        '14670.0': '25.0',
        '14672.0': '25.0',
        '14673.0': '25.0',
        '14674.0': '77.0',
        '14675.0': '189.0',
        '14676.0': '189.0',
        '14678.0': '71.0',
        '14679.0': '25.0',
        '14681.0': '189.0',
        '14682.0': '25.0',
        '14683.0': '25.0',
        '14685.0': '189.0',
        '14686.0': '189.0',
        '14688.0': '189.0',
        '14689.0': '71.0',
        '14691.0': '77.0',
        '14694.0': '96.0',
        '14695.0': '342.0',
        '14698.0': '342.0',
        '14704.0': '128.0',
        '14705.0': '342.0',
        '14706.0': '96.0',
        '14707.0': '342.0',
        '14708.0': '342.0',
        '14710.0': '342.0',
        '14713.0': '342.0',
        '14714.0': '342.0',
        '14717.0': '342.0',
        '14719.0': '342.0',
        '14723.0': '77.0',
        '14724.0': '96.0',
        '14726.0': '342.0',
        '14728.0': '342.0',
        '14729.0': '342.0',
        '14731.0': '342.0',
        '14732.0': '342.0',
        '14733.0': '342.0',
        '14735.0': '342.0',
        '14736.0': '342.0',
        '14737.0': '342.0',
        '14739.0': '342.0',
        '14740.0': '342.0',
        '14742.0': '342.0',
        '14744.0': '342.0',
        '14745.0': '342.0',
        '14746.0': '342.0',
        '14749.0': '342.0',
        '14750.0': '342.0',
        '14756.0': '342.0',
        '14759.0': '342.0',
        '14762.0': '342.0',
        '14956.0': '152.0',
        '1505.0': '66.0',
        '1506.0': '66.0',
        '1508.0': '66.0',
        '151.0': '103.0',
        '1510.0': '66.0',
        '15198.0': '71.0',
        '15205.0': '71.0',
        '15206.0': '71.0',
        '15207.0': '71.0',
        '15208.0': '71.0',
        '15213.0': '249.0',
        '15215.0': '60.0',
        '15216.0': '196.0',
        '15224.0': '120.0',
        '15228.0': '136.0',
        '15229.0': '136.0',
        '15232.0': '259.0',
        '15248.0': '23.0',
        '15268.0': '333.0',
        '15270.0': '333.0',
        '15271.0': '333.0',
        '15272.0': '333.0',
        '15273.0': '333.0',
        '15274.0': '333.0',
        '15277.0': '333.0',
        '15280.0': '333.0',
        '15282.0': '333.0',
        '15283.0': '333.0',
        '15284.0': '333.0',
        '15290.0': '333.0',
        '15291.0': '333.0',
        '15293.0': '92.0',
        '15341.0': '238.0',
        '15371.0': '263.0',
        '15376.0': '263.0',
        '1538.0': '66.0',
        '15418.0': '91.0',
        '15544.0': '142.0',
        '15549.0': '142.0',
        '15558.0': '142.0',
        '15562.0': '142.0',
        '15592.0': '339.0',
        '15593.0': '339.0',
        '15594.0': '339.0',
        '15595.0': '339.0',
        '15596.0': '339.0',
        '15608.0': '339.0',
        '15609.0': '339.0',
        '15610.0': '339.0',
        '15611.0': '339.0',
        '15613.0': '339.0',
        '15614.0': '339.0',
        '15637.0': '26.0',
        '15638.0': '26.0',
        '15642.0': '26.0',
        '15643.0': '26.0',
        '15645.0': '26.0',
        '15649.0': '26.0',
        '15680.0': '17.0',
        '15702.0': '368.0',
        '15703.0': '368.0',
        '15704.0': '368.0',
        '15705.0': '368.0',
        '15706.0': '368.0',
        '15707.0': '368.0',
        '15708.0': '368.0',
        '15709.0': '368.0',
        '15710.0': '368.0',
        '15711.0': '368.0',
        '15714.0': '368.0',
        '15716.0': '368.0',
        '15717.0': '368.0',
        '15718.0': '368.0',
        '15720.0': '368.0',
        '15721.0': '368.0',
        '15722.0': '368.0',
        '15723.0': '368.0',
        '15724.0': '368.0',
        '15725.0': '368.0',
        '15726.0': '368.0',
        '15727.0': '96.0',
        '15728.0': '368.0',
        '1573.0': '383.0',
        '15730.0': '368.0',
        '15731.0': '368.0',
        '15732.0': '368.0',
        '15733.0': '368.0',
        '15734.0': '368.0',
        '15735.0': '368.0',
        '1578.0': '383.0',
        '15861.0': '308.0',
        '15866.0': '308.0',
        '15867.0': '308.0',
        '15876.0': '48.0',
        '15895.0': '48.0',
        '15898.0': '48.0',
        '15900.0': '48.0',
        '15902.0': '48.0',
        '15904.0': '48.0',
        '15905.0': '48.0',
        '1593.0': '52.0',
        '1594.0': '52.0',
        '1595.0': '52.0',
        '15956.0': '310.0',
        '15957.0': '310.0',
        '1596.0': '52.0',
        '15960.0': '310.0',
        '15962.0': '310.0',
        '1597.0': '52.0',
        '15980.0': '263.0',
        '15989.0': '278.0',
        '15990.0': '278.0',
        '15997.0': '278.0',
        '1602.0': '52.0',
        '16035.0': '218.0',
        '16037.0': '218.0',
        '16054.0': '218.0',
        '16056.0': '218.0',
        '16057.0': '218.0',
        '16067.0': '260.0',
        '16069.0': '260.0',
        '16070.0': '71.0',
        '16075.0': '77.0',
        '1609.0': '52.0',
        '16099.0': '128.0',
        '16100.0': '128.0',
        '16101.0': '128.0',
        '16102.0': '128.0',
        '16104.0': '128.0',
        '16105.0': '128.0',
        '16106.0': '128.0',
        '16107.0': '128.0',
        '16108.0': '128.0',
        '16109.0': '128.0',
        '16110.0': '128.0',
        '16111.0': '128.0',
        '16112.0': '128.0',
        '16113.0': '128.0',
        '16115.0': '128.0',
        '16116.0': '327.0',
        '16118.0': '327.0',
        '16119.0': '327.0',
        '16121.0': '327.0',
        '16122.0': '327.0',
        '16123.0': '189.0',
        '16124.0': '327.0',
        '16125.0': '327.0',
        '16126.0': '327.0',
        '16127.0': '327.0',
        '16128.0': '128.0',
        '16129.0': '327.0',
        '16131.0': '327.0',
        '16132.0': '327.0',
        '16133.0': '327.0',
        '16142.0': '327.0',
        '16147.0': '236.0',
        '16148.0': '236.0',
        '16150.0': '236.0',
        '16152.0': '317.0',
        '16153.0': '317.0',
        '16155.0': '317.0',
        '16156.0': '317.0',
        '16157.0': '317.0',
        '16159.0': '317.0',
        '16163.0': '317.0',
        '16165.0': '317.0',
        '16167.0': '317.0',
        '16168.0': '317.0',
        '16169.0': '317.0',
        '16170.0': '317.0',
        '16171.0': '317.0',
        '1618.0': '175.0',
        '16194.0': '288.0',
        '16195.0': '288.0',
        '16196.0': '288.0',
        '16197.0': '288.0',
        '16199.0': '288.0',
        '16200.0': '288.0',
        '16201.0': '288.0',
        '16202.0': '288.0',
        '16203.0': '288.0',
        '16204.0': '288.0',
        '16205.0': '288.0',
        '16206.0': '288.0',
        '16207.0': '288.0',
        '16208.0': '288.0',
        '16209.0': '288.0',
        '16210.0': '288.0',
        '16211.0': '288.0',
        '16212.0': '288.0',
        '16213.0': '288.0',
        '16215.0': '288.0',
        '16216.0': '288.0',
        '16217.0': '288.0',
        '16220.0': '327.0',
        '16221.0': '327.0',
        '16222.0': '327.0',
        '16224.0': '327.0',
        '16227.0': '327.0',
        '16228.0': '327.0',
        '16229.0': '327.0',
        '16230.0': '327.0',
        '16231.0': '327.0',
        '16232.0': '327.0',
        '16233.0': '327.0',
        '16236.0': '327.0',
        '16237.0': '327.0',
        '16238.0': '327.0',
        '16239.0': '327.0',
        '16240.0': '327.0',
        '16246.0': '44.0',
        '16249.0': '44.0',
        '16252.0': '44.0',
        '16254.0': '44.0',
        '16255.0': '44.0',
        '16257.0': '44.0',
        '16259.0': '114.0',
        '16262.0': '44.0',
        '16263.0': '44.0',
        '16264.0': '44.0',
        '16265.0': '44.0',
        '16277.0': '100.0',
        '16278.0': '100.0',
        '16323.0': '291.0',
        '16332.0': '13.0',
        '16334.0': '13.0',
        '16378.0': '10.0',
        '16434.0': '0.0',
        '1644.0': '369.0',
        '1645.0': '369.0',
        '1646.0': '369.0',
        '16467.0': '132.0',
        '16470.0': '132.0',
        '1649.0': '369.0',
        '16511.0': '139.0',
        '16514.0': '139.0',
        '16515.0': '139.0',
        '16520.0': '139.0',
        '16528.0': '139.0',
        '16569.0': '143.0',
        '16582.0': '139.0',
        '16584.0': '139.0',
        '16585.0': '139.0',
        '16618.0': '119.0',
        '16779.0': '114.0',
        '16780.0': '114.0',
        '16781.0': '114.0',
        '16782.0': '384.0',
        '16783.0': '114.0',
        '16784.0': '114.0',
        '16786.0': '114.0',
        '16788.0': '114.0',
        '16789.0': '114.0',
        '16790.0': '114.0',
        '16791.0': '114.0',
        '16792.0': '196.0',
        '16793.0': '114.0',
        '16794.0': '114.0',
        '16795.0': '114.0',
        '16797.0': '249.0',
        '16801.0': '249.0',
        '16806.0': '249.0',
        '16807.0': '249.0',
        '16809.0': '249.0',
        '16811.0': '249.0',
        '16813.0': '249.0',
        '16815.0': '249.0',
        '16816.0': '249.0',
        '16824.0': '384.0',
        '16829.0': '384.0',
        '16839.0': '384.0',
        '16840.0': '384.0',
        '16846.0': '196.0',
        '1688.0': '398.0',
        '16884.0': '88.0',
        '16886.0': '88.0',
        '1695.0': '398.0',
        '16954.0': '291.0',
        '16984.0': '97.0',
        '16985.0': '97.0',
        '16986.0': '97.0',
        '16987.0': '97.0',
        '16990.0': '97.0',
        '16991.0': '97.0',
        '16992.0': '97.0',
        '16993.0': '97.0',
        '16994.0': '97.0',
        '16997.0': '97.0',
        '16999.0': '97.0',
        '1700.0': '398.0',
        '17020.0': '357.0',
        '17029.0': '108.0',
        '17031.0': '131.0',
        '17034.0': '357.0',
        '17037.0': '357.0',
        '17068.0': '366.0',
        '17076.0': '366.0',
        '17078.0': '274.0',
        '17095.0': '366.0',
        '1710.0': '70.0',
        '17121.0': '8.0',
        '17170.0': '312.0',
        '17181.0': '8.0',
        '17191.0': '8.0',
        '17193.0': '8.0',
        '17194.0': '277.0',
        '17201.0': '274.0',
        '17218.0': '343.0',
        '17252.0': '8.0',
        '17258.0': '318.0',
        '17260.0': '318.0',
        '17262.0': '318.0',
        '17264.0': '318.0',
        '17265.0': '318.0',
        '17266.0': '318.0',
        '17268.0': '318.0',
        '17270.0': '318.0',
        '17274.0': '318.0',
        '17294.0': '8.0',
        '17298.0': '259.0',
        '17356.0': '58.0',
        '17364.0': '58.0',
        '17369.0': '58.0',
        '17370.0': '58.0',
        '17426.0': '19.0',
        '17472.0': '215.0',
        '17475.0': '215.0',
        '17482.0': '215.0',
        '175.0': '249.0',
        '17534.0': '205.0',
        '17543.0': '220.0',
        '17546.0': '220.0',
        '17551.0': '220.0',
        '17554.0': '220.0',
        '17587.0': '364.0',
        '17588.0': '195.0',
        '17610.0': '92.0',
        '17667.0': '379.0',
        '17671.0': '265.0',
        '17680.0': '17.0',
        '17681.0': '17.0',
        '17682.0': '17.0',
        '17683.0': '17.0',
        '17687.0': '17.0',
        '17689.0': '17.0',
        '17690.0': '17.0',
        '17692.0': '17.0',
        '17694.0': '17.0',
        '17696.0': '17.0',
        '17707.0': '114.0',
        '17709.0': '114.0',
        '17712.0': '114.0',
        '17721.0': '114.0',
        '17723.0': '114.0',
        '17758.0': '143.0',
        '1777.0': '355.0',
        '178.0': '249.0',
        '1780.0': '355.0',
        '1782.0': '178.0',
        '1786.0': '355.0',
        '17872.0': '340.0',
        '18022.0': '26.0',
        '18024.0': '26.0',
        '18029.0': '26.0',
        '18030.0': '26.0',
        '18031.0': '26.0',
        '18032.0': '26.0',
        '18035.0': '26.0',
        '18036.0': '26.0',
        '18037.0': '26.0',
        '18038.0': '26.0',
        '18045.0': '26.0',
        '18048.0': '26.0',
        '18053.0': '26.0',
        '18054.0': '26.0',
        '18058.0': '26.0',
        '18059.0': '26.0',
        '18075.0': '381.0',
        '18077.0': '381.0',
        '18078.0': '381.0',
        '18088.0': '193.0',
        '18180.0': '348.0',
        '18256.0': '97.0',
        '18257.0': '97.0',
        '18258.0': '97.0',
        '18260.0': '97.0',
        '18261.0': '97.0',
        '18263.0': '97.0',
        '18265.0': '97.0',
        '18270.0': '97.0',
        '18276.0': '97.0',
        '18282.0': '97.0',
        '18289.0': '360.0',
        '18308.0': '274.0',
        '18311.0': '259.0',
        '18316.0': '35.0',
        '18327.0': '35.0',
        '18335.0': '371.0',
        '18381.0': '8.0',
        '18382.0': '8.0',
        '18391.0': '343.0',
        '18399.0': '274.0',
        '1841.0': '274.0',
        '18412.0': '343.0',
        '18413.0': '343.0',
        '18422.0': '343.0',
        '18423.0': '343.0',
        '18424.0': '274.0',
        '18438.0': '196.0',
        '18456.0': '274.0',
        '18472.0': '66.0',
        '18492.0': '153.0',
        '18510.0': '398.0',
        '18518.0': '398.0',
        '18520.0': '140.0',
        '18521.0': '140.0',
        '18559.0': '155.0',
        '18561.0': '285.0',
        '18562.0': '285.0',
        '18565.0': '155.0',
        '18567.0': '243.0',
        '18572.0': '301.0',
        '18593.0': '116.0',
        '18599.0': '243.0',
        '18603.0': '19.0',
        '18611.0': '354.0',
        '18612.0': '354.0',
        '18633.0': '116.0',
        '18634.0': '302.0',
        '18638.0': '232.0',
        '18641.0': '369.0',
        '18642.0': '369.0',
        '18645.0': '369.0',
        '18646.0': '369.0',
        '18649.0': '70.0',
        '18650.0': '70.0',
        '18651.0': '70.0',
        '18652.0': '70.0',
        '18654.0': '70.0',
        '18655.0': '70.0',
        '18661.0': '70.0',
        '18662.0': '70.0',
        '1869.0': '205.0',
        '18698.0': '379.0',
        '18708.0': '220.0',
        '18746.0': '61.0',
        '18748.0': '61.0',
        '18765.0': '155.0',
        '18794.0': '83.0',
        '18853.0': '301.0',
        '18854.0': '301.0',
        '18855.0': '301.0',
        '18857.0': '301.0',
        '18858.0': '301.0',
        '18859.0': '301.0',
        '18861.0': '301.0',
        '18863.0': '301.0',
        '18865.0': '301.0',
        '18866.0': '301.0',
        '18868.0': '301.0',
        '1889.0': '259.0',
        '1890.0': '259.0',
        '18933.0': '161.0',
        '18934.0': '161.0',
        '18936.0': '161.0',
        '18937.0': '161.0',
        '18939.0': '161.0',
        '18944.0': '161.0',
        '1895.0': '259.0',
        '18956.0': '260.0',
        '18957.0': '260.0',
        '18958.0': '260.0',
        '18959.0': '260.0',
        '18960.0': '260.0',
        '18961.0': '260.0',
        '18962.0': '260.0',
        '18964.0': '260.0',
        '18965.0': '260.0',
        '18967.0': '260.0',
        '18969.0': '260.0',
        '18971.0': '260.0',
        '18973.0': '260.0',
        '18977.0': '260.0',
        '18979.0': '260.0',
        '18982.0': '260.0',
        '18984.0': '260.0',
        '18988.0': '260.0',
        '18989.0': '293.0',
        '18990.0': '293.0',
        '18991.0': '293.0',
        '18992.0': '293.0',
        '18993.0': '293.0',
        '18994.0': '293.0',
        '18995.0': '293.0',
        '18996.0': '293.0',
        '18997.0': '293.0',
        '18998.0': '293.0',
        '18999.0': '293.0',
        '19000.0': '293.0',
        '19002.0': '293.0',
        '19004.0': '293.0',
        '19008.0': '293.0',
        '19009.0': '293.0',
        '19010.0': '293.0',
        '19011.0': '293.0',
        '19017.0': '60.0',
        '19020.0': '60.0',
        '19029.0': '60.0',
        '19040.0': '60.0',
        '19044.0': '153.0',
        '19045.0': '153.0',
        '19051.0': '153.0',
        '19053.0': '153.0',
        '19057.0': '153.0',
        '19097.0': '341.0',
        '19171.0': '394.0',
        '19176.0': '394.0',
        '19186.0': '394.0',
        '19225.0': '119.0',
        '19331.0': '338.0',
        '19348.0': '20.0',
        '19384.0': '259.0',
        '19392.0': '259.0',
        '19471.0': '187.0',
        '19476.0': '187.0',
        '19477.0': '187.0',
        '19479.0': '187.0',
        '19482.0': '187.0',
        '19755.0': '323.0',
        '19756.0': '323.0',
        '19758.0': '323.0',
        '19759.0': '323.0',
        '19806.0': '97.0',
        '19817.0': '254.0',
        '19829.0': '17.0',
        '19834.0': '17.0',
        '19860.0': '254.0',
        '19862.0': '254.0',
        '199.0': '183.0',
        '19923.0': '297.0',
        '19972.0': '88.0',
        '20009.0': '48.0',
        '20061.0': '131.0',
        '20065.0': '41.0',
        '20067.0': '183.0',
        '20069.0': '336.0',
        '20080.0': '375.0',
        '20138.0': '167.0',
        '20141.0': '37.0',
        '20168.0': '92.0',
        '20172.0': '40.0',
        '20200.0': '249.0',
        '20205.0': '323.0',
        '20206.0': '323.0',
        '20207.0': '323.0',
        '20208.0': '323.0',
        '20209.0': '323.0',
        '20212.0': '323.0',
        '20213.0': '66.0',
        '20226.0': '196.0',
        '20239.0': '144.0',
        '20243.0': '342.0',
        '20267.0': '63.0',
        '20268.0': '63.0',
        '20271.0': '17.0',
        '20279.0': '17.0',
        '20286.0': '29.0',
        '20288.0': '342.0',
        '20332.0': '40.0',
        '2034.0': '379.0',
        '2036.0': '379.0',
        '2037.0': '379.0',
        '2039.0': '20.0',
        '20391.0': '120.0',
        '20395.0': '354.0',
        '20461.0': '333.0',
        '20483.0': '140.0',
        '20486.0': '140.0',
        '20488.0': '140.0',
        '20497.0': '267.0',
        '20526.0': '354.0',
        '20558.0': '187.0',
        '20578.0': '52.0',
        '20608.0': '63.0',
        '20609.0': '63.0',
        '20610.0': '63.0',
        '20612.0': '63.0',
        '20614.0': '63.0',
        '20615.0': '63.0',
        '20618.0': '63.0',
        '20620.0': '370.0',
        '20695.0': '297.0',
        '20704.0': '126.0',
        '20728.0': '323.0',
        '20733.0': '132.0',
        '20744.0': '369.0',
        '20761.0': '214.0',
        '20776.0': '155.0',
        '20826.0': '349.0',
        '20842.0': '285.0',
        '20843.0': '156.0',
        '20844.0': '156.0',
        '20845.0': '156.0',
        '20855.0': '381.0',
        '20877.0': '153.0',
        '209.0': '183.0',
        '20900.0': '147.0',
        '20950.0': '189.0',
        '20960.0': '348.0',
        '20964.0': '250.0',
        '21000.0': '178.0',
        '21049.0': '342.0',
        '21062.0': '153.0',
        '21065.0': '369.0',
        '21097.0': '211.0',
        '21144.0': '164.0',
        '21193.0': '23.0',
        '2125.0': '220.0',
        '21271.0': '126.0',
        '21276.0': '372.0',
        '21283.0': '372.0',
        '21284.0': '372.0',
        '21286.0': '372.0',
        '21288.0': '372.0',
        '21292.0': '372.0',
        '21298.0': '372.0',
        '213.0': '183.0',
        '21302.0': '372.0',
        '21305.0': '372.0',
        '21311.0': '155.0',
        '21317.0': '37.0',
        '21318.0': '37.0',
        '21319.0': '92.0',
        '21320.0': '92.0',
        '21323.0': '92.0',
        '21326.0': '370.0',
        '21344.0': '308.0',
        '21346.0': '308.0',
        '21357.0': '197.0',
        '21370.0': '68.0',
        '2139.0': '238.0',
        '214.0': '183.0',
        '21408.0': '258.0',
        '21438.0': '156.0',
        '21442.0': '370.0',
        '21447.0': '202.0',
        '21461.0': '26.0',
        '21463.0': '317.0',
        '21466.0': '317.0',
        '21467.0': '317.0',
        '21470.0': '317.0',
        '21471.0': '317.0',
        '21473.0': '317.0',
        '21475.0': '317.0',
        '21476.0': '317.0',
        '21479.0': '317.0',
        '21480.0': '317.0',
        '21505.0': '92.0',
        '21507.0': '92.0',
        '21510.0': '92.0',
        '21511.0': '92.0',
        '21512.0': '92.0',
        '21513.0': '92.0',
        '21514.0': '92.0',
        '21515.0': '92.0',
        '21516.0': '92.0',
        '21517.0': '92.0',
        '21518.0': '92.0',
        '21519.0': '92.0',
        '21520.0': '92.0',
        '21521.0': '92.0',
        '21523.0': '92.0',
        '21529.0': '92.0',
        '21532.0': '92.0',
        '21533.0': '92.0',
        '21534.0': '92.0',
        '21651.0': '128.0',
        '21652.0': '128.0',
        '21653.0': '128.0',
        '21654.0': '128.0',
        '21656.0': '128.0',
        '21657.0': '128.0',
        '21661.0': '189.0',
        '21662.0': '189.0',
        '21664.0': '189.0',
        '21665.0': '189.0',
        '21666.0': '189.0',
        '21667.0': '189.0',
        '21669.0': '189.0',
        '21670.0': '189.0',
        '21672.0': '25.0',
        '21675.0': '25.0',
        '21676.0': '189.0',
        '21690.0': '96.0',
        '21691.0': '96.0',
        '21693.0': '96.0',
        '21694.0': '96.0',
        '21697.0': '96.0',
        '21699.0': '136.0',
        '2170.0': '234.0',
        '21705.0': '96.0',
        '21706.0': '96.0',
        '21714.0': '96.0',
        '21715.0': '198.0',
        '21716.0': '198.0',
        '21717.0': '198.0',
        '21718.0': '198.0',
        '21719.0': '198.0',
        '21720.0': '198.0',
        '21721.0': '198.0',
        '21722.0': '198.0',
        '21727.0': '198.0',
        '21734.0': '198.0',
        '21745.0': '77.0',
        '21746.0': '77.0',
        '21747.0': '77.0',
        '21749.0': '77.0',
        '21750.0': '77.0',
        '21752.0': '77.0',
        '21754.0': '77.0',
        '21758.0': '77.0',
        '21763.0': '77.0',
        '21769.0': '77.0',
        '21770.0': '77.0',
        '21771.0': '77.0',
        '21772.0': '77.0',
        '21773.0': '77.0',
        '21778.0': '77.0',
        '21781.0': '77.0',
        '21782.0': '77.0',
        '21787.0': '257.0',
        '21788.0': '257.0',
        '21789.0': '257.0',
        '21790.0': '257.0',
        '21791.0': '257.0',
        '21792.0': '257.0',
        '21793.0': '257.0',
        '21794.0': '257.0',
        '21796.0': '257.0',
        '21797.0': '257.0',
        '21798.0': '257.0',
        '21799.0': '257.0',
        '21800.0': '257.0',
        '21801.0': '257.0',
        '21802.0': '257.0',
        '21805.0': '257.0',
        '21806.0': '77.0',
        '21819.0': '77.0',
        '21823.0': '44.0',
        '21824.0': '44.0',
        '21826.0': '44.0',
        '21827.0': '44.0',
        '21828.0': '44.0',
        '21832.0': '44.0',
        '21834.0': '44.0',
        '21837.0': '44.0',
        '21838.0': '44.0',
        '21844.0': '249.0',
        '21851.0': '44.0',
        '21857.0': '105.0',
        '21917.0': '250.0',
        '21918.0': '250.0',
        '21920.0': '140.0',
        '21932.0': '140.0',
        '21938.0': '140.0',
        '21978.0': '267.0',
        '21998.0': '267.0',
        '22014.0': '159.0',
        '22165.0': '52.0',
        '22193.0': '160.0',
        '22194.0': '160.0',
        '22232.0': '118.0',
        '22246.0': '118.0',
        '22251.0': '369.0',
        '22253.0': '369.0',
        '22263.0': '120.0',
        '22285.0': '120.0',
        '22291.0': '120.0',
        '22397.0': '137.0',
        '22403.0': '108.0',
        '22411.0': '398.0',
        '22426.0': '105.0',
        '22432.0': '70.0',
        '22438.0': '272.0',
        '22440.0': '272.0',
        '22544.0': '107.0',
        '22545.0': '107.0',
        '22546.0': '107.0',
        '22547.0': '107.0',
        '22548.0': '107.0',
        '22549.0': '107.0',
        '22550.0': '107.0',
        '22551.0': '107.0',
        '22552.0': '107.0',
        '22553.0': '107.0',
        '22554.0': '107.0',
        '22555.0': '107.0',
        '22561.0': '107.0',
        '22562.0': '107.0',
        '22563.0': '107.0',
        '22564.0': '107.0',
        '22566.0': '107.0',
        '22570.0': '107.0',
        '22572.0': '107.0',
        '22573.0': '107.0',
        '22580.0': '107.0',
        '22581.0': '107.0',
        '22585.0': '107.0',
        '22588.0': '107.0',
        '22589.0': '107.0',
        '22590.0': '107.0',
        '22592.0': '107.0',
        '22596.0': '107.0',
        '22598.0': '107.0',
        '22630.0': '163.0',
        '22642.0': '163.0',
        '22657.0': '178.0',
        '22659.0': '178.0',
        '22670.0': '178.0',
        '22693.0': '113.0',
        '22694.0': '113.0',
        '22696.0': '113.0',
        '22700.0': '113.0',
        '22704.0': '113.0',
        '22717.0': '274.0',
        '22728.0': '265.0',
        '22729.0': '343.0',
        '22751.0': '191.0',
        '2277.0': '61.0',
        '22782.0': '277.0',
        '22791.0': '277.0',
        '2283.0': '61.0',
        '22878.0': '274.0',
        '22880.0': '274.0',
        '22881.0': '274.0',
        '22886.0': '274.0',
        '22888.0': '274.0',
        '22889.0': '274.0',
        '2289.0': '61.0',
        '22890.0': '274.0',
        '22896.0': '193.0',
        '22898.0': '193.0',
        '22929.0': '46.0',
        '2301.0': '61.0',
        '23024.0': '220.0',
        '23030.0': '220.0',
        '23034.0': '220.0',
        '23045.0': '220.0',
        '23048.0': '220.0',
        '23052.0': '352.0',
        '23053.0': '352.0',
        '23057.0': '352.0',
        '23061.0': '352.0',
        '23074.0': '352.0',
        '23078.0': '352.0',
        '23099.0': '155.0',
        '2315.0': '243.0',
        '23192.0': '233.0',
        '23193.0': '233.0',
        '23194.0': '233.0',
        '23195.0': '233.0',
        '23196.0': '233.0',
        '23197.0': '233.0',
        '23198.0': '233.0',
        '23199.0': '233.0',
        '23203.0': '233.0',
        '23204.0': '233.0',
        '23205.0': '233.0',
        '23206.0': '233.0',
        '23207.0': '233.0',
        '23208.0': '233.0',
        '23209.0': '233.0',
        '23210.0': '233.0',
        '23213.0': '233.0',
        '23214.0': '233.0',
        '23215.0': '233.0',
        '23219.0': '233.0',
        '23220.0': '233.0',
        '23221.0': '233.0',
        '23231.0': '63.0',
        '23233.0': '63.0',
        '23234.0': '63.0',
        '23239.0': '29.0',
        '23270.0': '74.0',
        '23271.0': '74.0',
        '23274.0': '74.0',
        '23285.0': '74.0',
        '23293.0': '74.0',
        '23294.0': '74.0',
        '23298.0': '370.0',
        '23299.0': '370.0',
        '23301.0': '370.0',
        '23309.0': '370.0',
        '2354.0': '186.0',
        '23556.0': '254.0',
        '23558.0': '254.0',
        '23562.0': '254.0',
        '23565.0': '254.0',
        '23579.0': '254.0',
        '23580.0': '254.0',
        '23624.0': '95.0',
        '2372.0': '338.0',
        '2373.0': '338.0',
        '23733.0': '297.0',
        '23735.0': '83.0',
        '23741.0': '297.0',
        '23745.0': '83.0',
        '23752.0': '297.0',
        '23754.0': '297.0',
        '23759.0': '83.0',
        '23762.0': '83.0',
        '2382.0': '258.0',
        '23878.0': '350.0',
        '23885.0': '350.0',
        '23886.0': '350.0',
        '23898.0': '350.0',
        '23905.0': '249.0',
        '2392.0': '258.0',
        '23947.0': '132.0',
        '23958.0': '196.0',
        '2396.0': '230.0',
        '23967.0': '383.0',
        '23969.0': '383.0',
        '23970.0': '136.0',
        '23985.0': '211.0',
        '23994.0': '153.0',
        '2400.0': '195.0',
        '2401.0': '195.0',
        '2405.0': '195.0',
        '2408.0': '155.0',
        '2411.0': '195.0',
        '2412.0': '195.0',
        '2487.0': '92.0',
        '2489.0': '92.0',
        '2490.0': '92.0',
        '2491.0': '92.0',
        '2492.0': '92.0',
        '2494.0': '92.0',
        '2496.0': '92.0',
        '2497.0': '92.0',
        '2500.0': '29.0',
        '2501.0': '29.0',
        '2503.0': '29.0',
        '2504.0': '92.0',
        '2511.0': '387.0',
        '2550.0': '29.0',
        '2553.0': '29.0',
        '2558.0': '29.0',
        '2560.0': '155.0',
        '2563.0': '29.0',
        '2567.0': '37.0',
        '2571.0': '29.0',
        '2578.0': '387.0',
        '2579.0': '155.0',
        '2581.0': '155.0',
        '2582.0': '155.0',
        '2584.0': '155.0',
        '2612.0': '243.0',
        '2618.0': '21.0',
        '2620.0': '243.0',
        '2624.0': '243.0',
        '2636.0': '379.0',
        '2671.0': '155.0',
        '2680.0': '155.0',
        '2681.0': '350.0',
        '2689.0': '301.0',
        '2692.0': '301.0',
        '2700.0': '301.0',
        '2723.0': '329.0',
        '2790.0': '21.0',
        '2817.0': '136.0',
        '2820.0': '136.0',
        '2823.0': '136.0',
        '2824.0': '136.0',
        '2831.0': '136.0',
        '2832.0': '136.0',
        '286.0': '136.0',
        '287.0': '136.0',
        '297.0': '288.0',
        '3007.0': '24.0',
        '301.0': '136.0',
        '303.0': '136.0',
        '304.0': '136.0',
        '306.0': '136.0',
        '3155.0': '155.0',
        '3214.0': '241.0',
        '3267.0': '211.0',
        '3268.0': '211.0',
        '3277.0': '211.0',
        '3314.0': '40.0',
        '3315.0': '40.0',
        '3325.0': '40.0',
        '333.0': '196.0',
        '3409.0': '37.0',
        '3410.0': '37.0',
        '3414.0': '37.0',
        '3415.0': '37.0',
        '3465.0': '189.0',
        '3466.0': '189.0',
        '3467.0': '189.0',
        '3468.0': '189.0',
        '3470.0': '189.0',
        '3474.0': '189.0',
        '3475.0': '189.0',
        '3476.0': '189.0',
        '3477.0': '189.0',
        '3480.0': '189.0',
        '3515.0': '168.0',
        '3523.0': '168.0',
        '3557.0': '66.0',
        '363.0': '23.0',
        '3686.0': '253.0',
        '3691.0': '253.0',
        '3704.0': '120.0',
        '3707.0': '120.0',
        '3740.0': '19.0',
        '3746.0': '354.0',
        '3774.0': '153.0',
        '3775.0': '153.0',
        '3783.0': '153.0',
        '3967.0': '178.0',
        '3982.0': '70.0',
        '3990.0': '70.0',
        '3992.0': '70.0',
        '3997.0': '70.0',
        '4008.0': '70.0',
        '4084.0': '20.0',
        '4222.0': '318.0',
        '4223.0': '318.0',
        '4229.0': '326.0',
        '4242.0': '338.0',
        '4296.0': '20.0',
        '4538.0': '109.0',
        '4539.0': '109.0',
        '4627.0': '387.0',
        '4628.0': '63.0',
        '4630.0': '21.0',
        '4631.0': '21.0',
        '4635.0': '21.0',
        '4639.0': '21.0',
        '4722.0': '29.0',
        '4724.0': '29.0',
        '4726.0': '29.0',
        '4732.0': '387.0',
        '4947.0': '34.0',
        '4950.0': '34.0',
        '4951.0': '34.0',
        '4997.0': '95.0',
        '5089.0': '162.0',
        '5116.0': '258.0',
        '5117.0': '387.0',
        '5133.0': '258.0',
        '5199.0': '272.0',
        '5200.0': '272.0',
        '5203.0': '272.0',
        '5207.0': '272.0',
        '5208.0': '272.0',
        '5221.0': '272.0',
        '5243.0': '272.0',
        '5249.0': '302.0',
        '5299.0': '135.0',
        '5317.0': '257.0',
        '5319.0': '257.0',
        '5321.0': '257.0',
        '5339.0': '267.0',
        '5433.0': '10.0',
        '5436.0': '10.0',
        '5439.0': '10.0',
        '5441.0': '10.0',
        '5461.0': '28.0',
        '5467.0': '118.0',
        '5509.0': '168.0',
        '5511.0': '118.0',
        '5518.0': '118.0',
        '5606.0': '108.0',
        '5616.0': '116.0',
        '5637.0': '116.0',
        '5664.0': '283.0',
        '5665.0': '131.0',
        '5666.0': '108.0',
        '5667.0': '337.0',
        '5672.0': '86.0',
        '5673.0': '169.0',
        '5675.0': '302.0',
        '5676.0': '252.0',
        '5677.0': '41.0',
        '5678.0': '18.0',
        '5679.0': '174.0',
        '5682.0': '116.0',
        '5684.0': '31.0',
        '5711.0': '131.0',
        '5720.0': '169.0',
        '5757.0': '262.0',
        '5767.0': '169.0',
        '5777.0': '137.0',
        '5982.0': '174.0',
        '6329.0': '174.0',
        '6340.0': '131.0',
        '6343.0': '105.0',
        '6457.0': '375.0',
        '6458.0': '86.0',
        '6460.0': '105.0',
        '6462.0': '302.0',
        '6493.0': '86.0',
        '6542.0': '137.0',
        '6765.0': '353.0',
        '678.0': '195.0',
        '680.0': '155.0',
        '684.0': '387.0',
        '6850.0': '86.0',
        '688.0': '387.0',
        '6882.0': '287.0',
        '6883.0': '287.0',
        '6884.0': '287.0',
        '6887.0': '287.0',
        '6888.0': '287.0',
        '6890.0': '287.0',
        '6898.0': '299.0',
        '6902.0': '299.0',
        '694.0': '155.0',
        '698.0': '387.0',
        '7067.0': '115.0',
        '7253.0': '378.0',
        '7257.0': '378.0',
        '7258.0': '378.0',
        '7260.0': '378.0',
        '7277.0': '236.0',
        '7314.0': '92.0',
        '7318.0': '92.0',
        '7320.0': '92.0',
        '7326.0': '92.0',
        '7367.0': '44.0',
        '7370.0': '44.0',
        '7371.0': '44.0',
        '7372.0': '44.0',
        '7374.0': '44.0',
        '7375.0': '257.0',
        '7376.0': '44.0',
        '7384.0': '236.0',
        '739.0': '147.0',
        '7393.0': '25.0',
        '7394.0': '25.0',
        '7399.0': '25.0',
        '740.0': '147.0',
        '741.0': '147.0',
        '743.0': '147.0',
        '745.0': '387.0',
        '7485.0': '354.0',
        '7489.0': '354.0',
        '7497.0': '7.0',
        '7524.0': '323.0',
        '7525.0': '323.0',
        '7527.0': '323.0',
        '7531.0': '323.0',
        '7534.0': '323.0',
        '7538.0': '44.0',
        '754.0': '387.0',
        '7550.0': '10.0',
        '7576.0': '354.0',
        '759.0': '229.0',
        '7630.0': '369.0',
        '7631.0': '369.0',
        '7636.0': '369.0',
        '7639.0': '369.0',
        '7654.0': '178.0',
        '772.0': '153.0',
        '7727.0': '70.0',
        '7729.0': '70.0',
        '7731.0': '70.0',
        '775.0': '379.0',
        '7769.0': '76.0',
        '7774.0': '274.0',
        '7799.0': '343.0',
        '7856.0': '379.0',
        '7941.0': '149.0',
        '796.0': '387.0',
        '8122.0': '63.0',
        '8150.0': '233.0',
        '8190.0': '17.0',
        '8452.0': '213.0',
        '8454.0': '213.0',
        '8457.0': '213.0',
        '8460.0': '213.0',
        '8616.0': '297.0',
        '8617.0': '297.0',
        '8677.0': '302.0',
        '8842.0': '333.0',
        '8849.0': '333.0',
        '8856.0': '92.0',
        '8868.0': '92.0',
        '8932.0': '308.0',
        '8936.0': '308.0',
        '8937.0': '308.0',
        '8953.0': '44.0',
        '8954.0': '44.0',
        '8956.0': '44.0',
        '8958.0': '44.0',
        '8965.0': '44.0',
        '8973.0': '249.0',
        '9068.0': '323.0',
        '9069.0': '323.0',
        '9232.0': '156.0',
        '9240.0': '156.0',
        '9247.0': '287.0',
        '9252.0': '287.0',
        '9254.0': '155.0',
        '9261.0': '287.0',
        '93.0': '105.0',
        '9328.0': '237.0',
        '9330.0': '237.0',
        '9331.0': '237.0',
        '9332.0': '237.0',
        '9344.0': '187.0',
        '9345.0': '187.0',
        '9352.0': '187.0',
        '9367.0': '187.0',
        '9450.0': '202.0',
        '9519.0': '167.0',
        '9528.0': '167.0',
        '9623.0': '314.0',
        '966.0': '71.0',
        '9660.0': '155.0',
        '967.0': '71.0',
        '968.0': '71.0',
        '970.0': '71.0',
        '971.0': '71.0',
        '972.0': '71.0',
        '973.0': '71.0',
        '9749.0': '147.0',
        '975.0': '71.0',
        '9750.0': '147.0',
        '9752.0': '147.0',
        '9754.0': '147.0',
        '9755.0': '147.0',
        '9756.0': '147.0',
        '9761.0': '147.0',
        '9762.0': '147.0',
        '9766.0': '147.0',
        '977.0': '71.0',
        '9788.0': '44.0',
        '9790.0': '154.0',
        '9793.0': '154.0',
        '9795.0': '154.0',
        '983.0': '301.0',
        '984.0': '301.0',
        '985.0': '301.0',
        '986.0': '301.0',
        '990.0': '301.0',
        '991.0': '301.0',
        '993.0': '301.0',
        '994.0': '301.0',
    },

    tasks: {
        '1.0': 'Resolve customer complaints regarding sales and service.',
        '10.0': 'Represent company at trade association meetings to promote products.',
        '1000.0': 'Prepare personnel forecast to project employment needs.',
        '1002.0': 'Develop, administer, and evaluate applicant tests.',
        '1003.0': 'Oversee the evaluation, classification, and rating of occupations and job positions.',
        '10039.0': 'Align and fit parts according to specifications, using jacks, turnbuckles, wedges, drift pins, pry bars, and hammers.',
        '10059.0': 'Observe temperature, humidity, pressure gauges, and product samples and adjust controls, such as thermostats and valves, to maintain prescribed operating conditions for specific stages.',
        '1014.0': 'Plan, develop, and provide training and staff development programs, using knowledge of the effectiveness of methods such as classroom training, demonstrations, on-the-job training, meetings, conferences, and workshops.',
        '10160.0': 'Record production data, and maintain production logs.',
        '10175.0': 'Set up, operate, or tend metal or plastic molding, casting, or coremaking machines to mold or cast metal or thermoplastic parts or products.',
        '10177.0': 'Turn valves and dials of machines to regulate pressure, temperature, and speed and feed rates, and to set cycle times.',
        '10189.0': 'Inventory and record quantities of materials and finished products, requisitioning additional supplies as necessary.',
        '10219.0': 'Record operational data, such as pressure readings, lengths of strokes, feed rates, or speeds.',
        '10295.0': 'Fit and align fabricated parts to be welded or assembled.',
        '10494.0': 'Monitor recording instruments, flowmeters, panel lights, or other indicators and listen for warning signals to verify conformity of process conditions.',
        '10495.0': 'Control or operate chemical processes or systems of machines, using panelboards, control boards, or semi-automatic equipment.',
        '10496.0': 'Record operating data, such as process conditions, test results, or instrument readings.',
        '10637.0': 'Maintain logs of working hours or of vehicle service or repair status, following applicable state and federal regulations.',
        '10644.0': 'Report vehicle defects, accidents, traffic violations, or damage to the vehicles.',
        '10671.0': 'Read switching instructions and daily car schedules to determine work to be performed, or receive orders from yard conductors.',
        '10672.0': 'Inspect the condition of stationary trains, rolling stock, and equipment.',
        '10674.0': 'Spot cars for loading and unloading at customer locations.',
        '10678.0': 'Receive, relay, and act upon instructions and inquiries from train operations and customer service center personnel.',
        '10680.0': 'Report arrival and departure times, train delays, work order completion, and time on duty.',
        '10691.0': 'Receive information regarding train or rail problems from dispatchers or from electronic monitoring devices.',
        '10697.0': 'Receive instructions from dispatchers regarding trains\' routes, timetables, and cargoes.',
        '10698.0': 'Review schedules, switching orders, way bills, and shipping records to obtain cargo loading and unloading information and to plan work.',
        '10702.0': 'Observe yard traffic to determine tracks available to accommodate inbound and outbound traffic.',
        '10704.0': 'Confirm routes and destination information for freight cars.',
        '10718.0': 'Provide customers with information about local roads or highways.',
        '10723.0': 'Maintain customer records and follow up periodically with telephone, mail, or personal reminders of services due.',
        '10835.0': 'Investigate complaints and suspected violations regarding illegal dumping, pollution, pesticides, product quality, or labeling laws.',
        '10838.0': 'Determine sampling locations and methods, and collect water or wastewater samples for analysis, preserving samples with appropriate containers and preservation methods.',
        '10842.0': 'Observe and record field conditions, gathering, interpreting, and reporting data such as flow meter readings and chemical levels.',
        '10902.0': 'Develop, test, or program new robots.',
        '10903.0': 'Perform complex calculations as part of the analysis and evaluation of data, using computers.',
        '10904.0': 'Describe and express observations and conclusions in mathematical terms.',
        '10905.0': 'Analyze data from research conducted to detect and measure physical phenomena.',
        '10906.0': 'Report experimental results by writing papers for scientific journals or by presenting information at scientific conferences.',
        '10907.0': 'Design computer simulations to model physical data so that it can be better understood.',
        '10908.0': 'Collaborate with other scientists in the design, development, and testing of experimental, industrial, or medical equipment, instrumentation, and procedures.',
        '10911.0': 'Develop theories and laws on the basis of observation and experiments, and apply these theories and laws to problems in areas such as nuclear energy, optics, and aerospace technology.',
        '10934.0': 'Pray and promote spirituality.',
        '10935.0': 'Read from sacred texts, such as the Bible, Torah, or Koran.',
        '10937.0': 'Organize and lead regular religious services.',
        '10938.0': 'Share information about religious issues by writing articles, giving speeches, or teaching.',
        '10940.0': 'Counsel individuals or groups concerning their spiritual, emotional, or personal needs.',
        '10944.0': 'Study and interpret religious laws, doctrines, or traditions.',
        '10951.0': 'Perform administrative duties, such as overseeing building management, ordering supplies, contracting for services or repairs, or supervising the work of staff members or volunteers.',
        '10956.0': 'Evaluate and grade examinations, assignments, or papers, and record grades.',
        '10958.0': 'Schedule and maintain regular office hours to meet with students.',
        '10959.0': 'Inform students of the procedures for completing and submitting class work, such as lab reports.',
        '10960.0': 'Prepare or proctor examinations.',
        '10963.0': 'Copy and distribute classroom materials.',
        '10967.0': 'Develop teaching materials, such as syllabi, visual aids, answer keys, supplementary notes, or course Web sites.',
        '10977.0': 'Research information requested by farmers.',
        '10989.0': 'Use materials such as pens and ink, watercolors, charcoal, oil, or computer software to create artwork.',
        '10990.0': 'Integrate and develop visual elements, such as line, space, mass, color, and perspective, to produce desired effects, such as the illustration of ideas, emotions, or moods.',
        '10991.0': 'Confer with clients, editors, writers, art directors, and other interested parties regarding the nature and content of artwork to be produced.',
        '10993.0': 'Maintain portfolios of artistic work to demonstrate styles, interests, and abilities.',
        '10994.0': 'Create finished art work as decoration, or to elucidate or substitute for spoken or written messages.',
        '10995.0': 'Cut, bend, laminate, arrange, and fasten individual or mixed raw and manufactured materials and products to form works of art.',
        '10996.0': 'Monitor events, trends, and other circumstances, research specific subject areas, attend art exhibitions, and read art publications to develop ideas and keep current on art world activities.',
        '10999.0': 'Create sketches, profiles, or likenesses of posed subjects or photographs, using any combination of freehand drawing, mechanical assembly kits, and computer imaging.',
        '11000.0': 'Create sculptures, statues, and other three-dimensional artwork by using abrasives and tools to shape, carve, and fabricate materials such as clay, stone, wood, or metal.',
        '11013.0': 'Provide entertainment at special events by performing activities such as drawing cartoons.',
        '1104.0': 'Greet and register guests.',
        '11046.0': 'Revise written material to meet personal standards and to satisfy needs of clients, publishers, directors, or producers.',
        '11047.0': 'Choose subject matter and suitable form to express personal feelings and experiences or ideas, or to narrate stories or events.',
        '11048.0': 'Plan project arrangements or outlines, and organize material accordingly.',
        '11049.0': 'Prepare works in appropriate format for publication, and send them to publishers or producers.',
        '1105.0': 'Answer inquiries pertaining to hotel policies and services, and resolve occupants\' complaints.',
        '11051.0': 'Write fiction or nonfiction prose, such as short stories, novels, biographies, articles, descriptive or critical analyses, and essays.',
        '11052.0': 'Develop factors such as themes, plots, characterizations, psychological analyses, historical environments, action, and dialogue to create material.',
        '11055.0': 'Write narrative, dramatic, lyric, or other types of poetry for publication.',
        '11057.0': 'Write words to fit musical compositions, including lyrics for operas, musical plays, and choral works.',
        '1106.0': 'Assign duties to workers, and schedule shifts.',
        '11060.0': 'Write humorous material for publication, or for performances such as comedy routines, gags, and comedy shows.',
        '1107.0': 'Coordinate front-office activities of hotels or motels, and resolve problems.',
        '1108.0': 'Participate in financial activities, such as the setting of room rates, the establishment of budgets, and the allocation of funds to departments.',
        '11102.0': 'Relay messages about emergencies, accidents, locations of crew and personnel, and fire hazard conditions.',
        '11104.0': 'Estimate sizes and characteristics of fires, and report findings to base camps by radio or telephone.',
        '1111.0': 'Manage and maintain temporary or permanent lodging facilities.',
        '1114.0': 'Show, rent, or assign accommodations.',
        '1115.0': 'Develop and implement policies and procedures for the operation of a department or establishment.',
        '11154.0': 'Take food orders and relay orders to kitchens or serving counters so they can be filled.',
        '11156.0': 'Remove trays and stack dishes for return to kitchen after meals are finished.',
        '11167.0': 'Perform serving, cleaning, or stocking duties in establishments, such as cafeterias or dining rooms, to facilitate customer service.',
        '11172.0': 'Locate items requested by customers.',
        '11178.0': 'Run cash registers.',
        '11189.0': 'Stock supplies, such as food or utensils, in serving stations, cupboards, refrigerators, or salad bars.',
        '1121.0': 'Perform marketing and public relations activities.',
        '1122.0': 'Organize and coordinate the work of staff and convention personnel for meetings to be held at a particular facility.',
        '11237.0': 'Answer customers\' questions about products, prices, availability, or credit terms.',
        '11238.0': 'Quote prices, credit terms, or other bid specifications.',
        '11239.0': 'Emphasize product features, based on analyses of customers\' needs and on technical knowledge of product capabilities and limitations.',
        '1124.0': 'Meet with clients to schedule and plan details of conventions, banquets, receptions and other functions.',
        '11240.0': 'Negotiate prices or terms of sales or service agreements.',
        '11241.0': 'Maintain customer records, using automated systems.',
        '11245.0': 'Collaborate with colleagues to exchange information, such as selling strategies or marketing information.',
        '11246.0': 'Prepare sales presentations or proposals to explain product specifications or applications.',
        '11251.0': 'Visit establishments to evaluate needs or to promote product or service sales.',
        '11252.0': 'Complete expense reports, sales reports, or other paperwork.',
        '11256.0': 'Provide feedback to product design teams so that products can be tailored to clients\' needs.',
        '1126.0': 'Book tickets for guests for local tours and attractions.',
        '11267.0': 'Visit establishments, such as pharmacies, to determine product sales.',
        '1129.0': 'Direct activities of professional and technical staff members and volunteers.',
        '11291.0': 'Verify and examine information and accuracy of loan application and closing documents.',
        '11292.0': 'Interview loan applicants to obtain personal and financial data and to assist in completing applications.',
        '11296.0': 'Record applications for loan and credit, loan information, and disbursements of funds, using computers.',
        '11300.0': 'Check value of customer collateral to be held as loan security.',
        '11301.0': 'Contact credit bureaus, employers, and other sources to check applicants\' credit and personal references.',
        '11304.0': 'Accept payment on accounts.',
        '11312.0': 'Verify readings in cases where consumption appears to be abnormal, and record possible reasons for fluctuations.',
        '11315.0': 'Answer customers\' questions about services and charges, or direct them to customer service centers.',
        '11369.0': 'Calculate figures, such as required amounts of labor or materials, manufacturing costs, or wages, using pricing schedules, adding machines, calculators, or computers.',
        '1137.0': 'Plan and administer budgets for programs, equipment, and support services.',
        '11406.0': 'Maintain logs of activities and completed work.',
        '11409.0': 'Resolve garbled or indecipherable messages, using cryptographic procedures and equipment.',
        '11431.0': 'Record information, such as personnel, production, or operational data on specified forms or reports.',
        '1151.0': 'Maintain and review computerized or manual records of purchased items, costs, deliveries, product performance, and inventories.',
        '11641.0': 'Enter information into computers to copy programs from one electronic component to another or to draw, modify, or store schematics.',
        '11710.0': 'Set machinery for proper performance, using computers.',
        '11761.0': 'Install, maintain, or repair security systems, alarm devices, or related equipment, following blueprints of electrical layouts and building plans.',
        '11821.0': 'Analyze test results, machine error messages, or information obtained from operators to diagnose equipment problems.',
        '11827.0': 'Enter codes and instructions to program computer-controlled machinery.',
        '11865.0': 'Play instruments to evaluate their sound quality and to locate any defects.',
        '11883.0': 'Test tubes and pickups in electronic amplifier units, and solder parts and connections as necessary.',
        '11890.0': 'Assemble bars onto percussion instruments.',
        '11893.0': 'Repair breaks in percussion instruments, such as drums and cymbals, using drill presses, power saws, glue, clamps, grinding wheels, or other hand tools.',
        '11896.0': 'Strike wood, fiberglass, or metal bars of instruments, and use tuned blocks, stroboscopes, or electronic tuners to evaluate tones made by instruments.',
        '11918.0': 'Measure or weigh flour or other ingredients to prepare batters, doughs, fillings, or icings, using scales or graduated containers.',
        '11937.0': 'Stop machines to remove finished workpieces or to change tooling, setup, or workpiece placement, according to required machining sequences.',
        '11941.0': 'Insert control instructions into machine control units to start operation.',
        '11944.0': 'Set up and operate computer-controlled machines or robots to perform one or more machine functions on metal or plastic workpieces.',
        '11946.0': 'Review program specifications or blueprints to determine and set machine operations and sequencing, finished workpiece dimensions, or numerical control sequences.',
        '11947.0': 'Monitor machine operation and control panel displays, and compare readings to specifications to detect malfunctions.',
        '11948.0': 'Control coolant systems.',
        '11961.0': 'Analyze job orders, drawings, blueprints, specifications, printed circuit board pattern films, and design data to calculate dimensions, tool selection, machine speeds, and feed rates.',
        '12175.0': 'Fit and study garments on customers to determine required alterations.',
        '12215.0': 'Attach parts or subassemblies together to form completed units, using glue, dowels, nails, screws, or clamps.',
        '12221.0': 'Program computers to operate machinery.',
        '1224.0': 'Organize registration of event participants.',
        '1229.0': 'Develop event topics and choose featured speakers.',
        '12301.0': 'Control, monitor, or operate equipment that regulates or distributes electricity or steam, using data obtained from instruments or computers.',
        '12304.0': 'Distribute or regulate the flow of power between entities, such as generating stations, substations, distribution lines, or users, keeping track of the status of circuits or connections.',
        '12306.0': 'Track conditions that could affect power needs, such as changes in the weather, and adjust equipment to meet any anticipated changes.',
        '12308.0': 'Calculate load estimates or equipment requirements to determine required control settings.',
        '12309.0': 'Record and compile operational data, such as chart or meter readings, power demands, or usage and operating times, using transmission system maps.',
        '12312.0': 'Implement energy schedules, including real-time transmission reservations or schedules.',
        '12336.0': 'Control or operate equipment in which chemical changes or reactions take place during the processing of industrial or consumer products.',
        '12400.0': 'Operate or tend machines to mix or blend any of a wide variety of materials, such as spices, dough batter, tobacco, fruit juices, chemicals, livestock feed, food products, color pigments, or explosive ingredients.',
        '1252.0': 'Evaluate customer records and recommend payment plans, based on earnings, savings data, payment history, and purchase activity.',
        '1256.0': 'Review individual or commercial customer files to identify and select delinquent accounts for collection.',
        '1257.0': 'Compare liquidity, profitability, and credit histories of establishments being evaluated with those of similar establishments in the same industries and geographic locations.',
        '1258.0': 'Consult with customers to resolve complaints and verify financial and credit transactions.',
        '12584.0': 'Monitor operation and adjust controls of processing machines and equipment to produce compositions with specific electronic properties, using computer terminals.',
        '1267.0': 'Correct errors by making appropriate changes and rechecking the program to ensure that the desired results are produced.',
        '1268.0': 'Conduct trial runs of programs and software applications to be sure they will produce the desired information and that the instructions are correct.',
        '1269.0': 'Compile and write documentation of program development and subsequent revisions, inserting comments in the coded instructions so others can understand the program.',
        '1270.0': 'Write, update, and maintain computer programs or software packages to handle specific jobs such as tracking inventory, storing or retrieving data, or controlling other equipment.',
        '1271.0': 'Consult with managerial, engineering, and technical personnel to clarify program intent, identify problems, and suggest changes.',
        '1272.0': 'Perform or direct revision, repair, or expansion of existing programs to increase operating efficiency or adapt to new requirements.',
        '12728.0': 'Issue landing and take-off authorizations or instructions.',
        '12729.0': 'Monitor or direct the movement of aircraft within an assigned air space or on the ground at airports to minimize delays and maximize safety.',
        '1273.0': 'Write, analyze, review, and rewrite programs, using workflow chart and diagram, and applying knowledge of computer capabilities, subject matter, and symbolic logic.',
        '12730.0': 'Monitor aircraft within a specific airspace, using radar, computer equipment, or visual references.',
        '12731.0': 'Inform pilots about nearby planes or potentially hazardous conditions, such as weather, speed and direction of wind, or visibility problems.',
        '12733.0': 'Alert airport emergency services in cases of emergency or when aircraft are experiencing difficulties.',
        '12739.0': 'Contact pilots by radio to provide meteorological, navigational, or other information.',
        '1274.0': 'Write or contribute to instructions or manuals to guide end users.',
        '12743.0': 'Compile information about flights from flight plans, pilot reports, radar, or observations.',
        '12746.0': 'Analyze factors such as weather reports, fuel requirements, or maps to determine air routes.',
        '1275.0': 'Investigate whether networks, workstations, the central processing unit of the system, or peripheral equipment are responding to a program\'s instructions.',
        '12755.0': 'Operate locomotives to transport freight or passengers between stations or to assemble or disassemble trains within rail yards.',
        '1276.0': 'Prepare detailed workflow charts and diagrams that describe input, output, and logical operation, and convert them into a series of instructions coded in a computer language.',
        '12760.0': 'Prepare reports regarding any problems encountered, such as accidents, signaling problems, unscheduled stops, or delays.',
        '12766.0': 'Drive and control rail-guided public transportation, such as subways, elevated trains, and electric-powered streetcars, trams, or trolleys, to transport passengers.',
        '12767.0': 'Monitor lights indicating obstructions or other trains ahead and watch for car and truck traffic at crossings to stay alert to potential hazards.',
        '1277.0': 'Perform systems analysis and programming tasks to maintain and control the use of computer systems software as a systems programmer.',
        '12770.0': 'Report delays, mechanical problems, and emergencies to supervisors or dispatchers, using radios.',
        '12771.0': 'Make announcements to passengers, such as notifications of upcoming stops or schedule delays.',
        '12773.0': 'Greet passengers, provide information, and answer questions concerning fares, schedules, transfers, and routings.',
        '1279.0': 'Assign, coordinate, and review work and activities of programming personnel.',
        '1280.0': 'Collaborate with computer manufacturers and other users to develop new programming methods.',
        '1282.0': 'Answer user inquiries regarding computer software or hardware operation to resolve problems.',
        '1283.0': 'Enter commands and observe system functioning to verify correct operations and detect errors.',
        '1285.0': 'Oversee the daily performance of computer systems.',
        '12878.0': 'Purchase, for further processing or for resale, farm products, such as milk, grains, or Christmas trees.',
        '12879.0': 'Negotiate contracts with farmers for the production or purchase of farm products.',
        '1288.0': 'Read technical manuals, confer with users, or conduct computer diagnostics to investigate and resolve problems or to provide technical assistance and support.',
        '1289.0': 'Confer with staff, users, and management to establish requirements for new systems or modifications.',
        '12891.0': 'Examine records, reports, or other documents to establish facts or detect discrepancies.',
        '12913.0': 'Prepare or interpret for clients information, such as investment performance reports, financial document summaries, or income projections.',
        '12929.0': 'Prepare reports or recommendations, based upon research outcomes.',
        '12930.0': 'Develop new methods to study the mechanisms of biological processes.',
        '12932.0': 'Share research findings by writing scientific articles or by making presentations at scientific conferences.',
        '12934.0': 'Develop or test new drugs or medications intended for commercial distribution.',
        '12935.0': 'Study the mutations in organisms that lead to cancer or other diseases.',
        '12936.0': 'Study spatial configurations of submicroscopic molecules, such as proteins, using x-rays or electron microscopes.',
        '12937.0': 'Study the chemistry of living processes, such as cell development, breathing and digestion, or living energy changes, such as growth, aging, or death.',
        '12938.0': 'Determine the three-dimensional structure of biological macromolecules.',
        '12940.0': 'Research the chemical effects of substances, such as drugs, serums, hormones, or food, on tissues or vital processes.',
        '12942.0': 'Develop methods to process, store, or use foods, drugs, or chemical compounds.',
        '12943.0': 'Investigate the nature, composition, or expression of genes or research how genetic engineering can impact these processes.',
        '12944.0': 'Study physical principles of living cells or organisms and their electrical or mechanical energy, applying methods and knowledge of mathematics, physics, chemistry, or biology.',
        '12946.0': 'Isolate, analyze, or synthesize vitamins, hormones, allergens, minerals, or enzymes and determine their effects on body functions.',
        '12947.0': 'Design or perform experiments with equipment, such as lasers, accelerators, or mass spectrometers.',
        '12951.0': 'Design or build laboratory equipment needed for special research projects.',
        '12952.0': 'Prepare, manipulate, and manage extensive databases.',
        '12953.0': 'Provide assistance with the preparation of project-related reports, manuscripts, and presentations.',
        '12955.0': 'Perform descriptive and multivariate statistical analyses of data, using computer software.',
        '12956.0': 'Verify the accuracy and validity of data entered in databases, correcting any errors.',
        '12957.0': 'Prepare tables, graphs, fact sheets, and written reports summarizing research results.',
        '12958.0': 'Edit and submit protocols and other required research documentation.',
        '12959.0': 'Develop and implement research quality control procedures.',
        '12960.0': 'Conduct internet-based and library research.',
        '12961.0': 'Present research findings to groups of people.',
        '12963.0': 'Design and create special programs for tasks such as statistical analysis and data entry and cleaning.',
        '12964.0': 'Code data in preparation for computer entry.',
        '12965.0': 'Provide assistance in the design of survey instruments such as questionnaires.',
        '12967.0': 'Administer standardized tests to research subjects, or interview them to collect research data.',
        '12968.0': 'Recruit and schedule research participants.',
        '12969.0': 'Track research participants, and perform any necessary follow-up tasks.',
        '12970.0': 'Allocate and manage laboratory space and resources.',
        '12973.0': 'Perform needs assessments or consult with clients to determine the types of research and information required.',
        '12983.0': 'Schedule special events, such as camps, conferences, meetings, seminars, or retreats.',
        '12988.0': 'Locate and distribute resources, such as periodicals or curricula, to enhance the effectiveness of educational programs.',
        '1299.0': 'Modify existing databases and database management systems or direct programmers and analysts to make changes.',
        '12993.0': 'Confer with disputants to clarify issues, identify underlying concerns, and develop an understanding of their respective needs and interests.',
        '12995.0': 'Set up appointments for parties to meet for mediation.',
        '12999.0': 'Prepare written opinions or decisions regarding cases.',
        '130.0': 'Prepare information regarding design, structure specifications, materials, color, equipment, estimated costs, or construction time.',
        '1300.0': 'Test programs or databases, correct errors, and make necessary modifications.',
        '13003.0': 'Evaluate information from documents, such as claim applications, birth or death certificates, or physician or employer records.',
        '13005.0': 'Research laws, regulations, policies, or precedent decisions to prepare for hearings.',
        '1301.0': 'Plan, coordinate, and implement security measures to safeguard information in computer files against accidental or unauthorized damage, modification or disclosure.',
        '1302.0': 'Approve, schedule, plan, and supervise the installation and testing of new products and improvements to computer systems, such as the installation of new databases.',
        '1303.0': 'Train users and answer questions.',
        '1305.0': 'Specify users and user access levels for each segment of database.',
        '1306.0': 'Develop data models describing data elements and how they are used, following procedures and using pen, template, or computer software.',
        '1309.0': 'Review procedures in database management system manuals to make changes to database.',
        '13096.0': 'Search computer databases, credit reports, public records, tax or legal filings, or other resources to locate persons or to compile information for investigations.',
        '13097.0': 'Obtain and analyze information on suspects, crimes, or disturbances to solve cases, to identify criminal activity, or to gather information for court cases.',
        '131.0': 'Consult with clients to determine functional or spatial requirements of structures.',
        '1311.0': 'Select and enter codes to monitor database performance and to create production databases.',
        '1312.0': 'Identify and evaluate industry trends in database systems to serve as a source of information and advice for upper management.',
        '1313.0': 'Write and code logical and physical database descriptions and specify identifiers of database to management system, or direct others in coding descriptions.',
        '13146.0': 'Requisition necessary supplies, equipment, or services.',
        '1315.0': 'Revise company definition of data as defined in data dictionary.',
        '13151.0': 'Recruit and hire staff members.',
        '13158.0': 'Arrange for tour or expedition details such as accommodations, transportation, equipment, and the availability of medical personnel.',
        '13160.0': 'Lead individuals or groups to tour site locations and describe points of interest.',
        '1317.0': 'Perform data backups and disaster recovery operations.',
        '1318.0': 'Maintain and administer computer networks and related computing environments, including computer hardware, systems software, applications software, and all configurations.',
        '1319.0': 'Plan, coordinate, and implement network security measures to protect data, software, and hardware.',
        '13191.0': 'Provide product samples, coupons, informational brochures, or other incentives to persuade people to buy products.',
        '13193.0': 'Record and report demonstration-related information, such as the number of questions asked by the audience or the number of coupons distributed.',
        '13198.0': 'Identify interested and qualified customers to provide them with additional information.',
        '1320.0': 'Operate master consoles to monitor the performance of computer systems and networks and to coordinate computer network access and use.',
        '13200.0': 'Prepare or alter presentation contents to target specific audiences.',
        '13201.0': 'Learn about competitors\' products or consumers\' interests or concerns to answer questions or provide more complete information.',
        '13203.0': 'Visit trade shows, stores, community organizations, or other venues to demonstrate products or services or to answer questions from potential customers.',
        '13206.0': 'Research or investigate products to be presented to prepare for demonstrations.',
        '1321.0': 'Perform routine network startup and shutdown procedures, and maintain control records.',
        '1322.0': 'Design, configure, and test computer hardware, networking software and operating system software.',
        '1323.0': 'Recommend changes to improve systems and network configurations, and determine hardware or software requirements related to such changes.',
        '13238.0': 'Develop prospect lists.',
        '1324.0': 'Confer with network users about solutions to existing system problems.',
        '13247.0': 'Operate telephone switchboards and systems to advance and complete connections, including those for local, long distance, pay telephone, mobile, person-to-person, and emergency calls.',
        '13248.0': 'Provide assistance for customers with special billing requests.',
        '1325.0': 'Monitor network performance to determine whether adjustments are needed and where changes will be needed in the future.',
        '13257.0': 'Update directory information.',
        '13258.0': 'Keep records of calls placed and received, and of related toll charges.',
        '1327.0': 'Load computer tapes and disks, and install software and printer paper or forms.',
        '1328.0': 'Gather data pertaining to customer needs, and use the information to identify, predict, interpret, and evaluate system and network requirements.',
        '1329.0': 'Analyze equipment performance records to determine the need for repair or replacement.',
        '1330.0': 'Maintain logs related to network functions, as well as maintenance and repair records.',
        '1333.0': 'Coordinate with vendors and with company personnel to facilitate purchases.',
        '1348.0': 'Design, implement, maintain, or improve electrical instruments, equipment, facilities, components, products, or systems for commercial, industrial, or domestic purposes.',
        '1349.0': 'Operate computer-assisted engineering or design software or equipment to perform engineering tasks.',
        '136.0': 'Integrate engineering elements into unified architectural designs.',
        '13695.0': 'Inspect core samples to determine nature of strata, or take samples to laboratories for analysis.',
        '13741.0': 'Tune or adjust equipment and instruments to obtain optimum visual or auditory reception, according to specifications, manuals, and drawings.',
        '13747.0': 'Position or mount speakers, and wire speakers to consoles.',
        '13837.0': 'Collect coins and bills from machines, prepare invoices, and settle accounts with concessionaires.',
        '13843.0': 'Record transaction information on forms or logs, and notify designated personnel of discrepancies.',
        '1409.0': 'Schedule deliveries based on production forecasts, material substitutions, storage and handling facilities, and maintenance requirements.',
        '1420.0': 'Conduct research that tests or analyzes the feasibility, design, operation, or performance of equipment, components, or systems.',
        '1447.0': 'Develop detailed design drawings and specifications for mechanical equipment, dies, tools, and controls, using computer-assisted drafting (CAD) equipment.',
        '1450.0': 'Compute mathematical formulas to develop and design detailed specifications for components or machinery, using computer-assisted equipment.',
        '1451.0': 'Position instructions and comments onto drawings.',
        '1452.0': 'Modify and revise designs to correct operating deficiencies or to reduce production problems.',
        '1455.0': 'Lay out and draw schematic, orthographic, or angle views to depict functional relationships of components, assemblies, systems, and machines.',
        '1461.0': 'Calculate dimensions, square footage, profile and component specifications, and material quantities, using calculator or computer.',
        '14613.0': 'Turn valves and start pumps to start or regulate flows of substances such as gases, liquids, slurries, or powdered materials.',
        '14623.0': 'Analyze problems to develop solutions involving computer hardware and software.',
        '14624.0': 'Assign or schedule tasks to meet work priorities and goals.',
        '14626.0': 'Apply theoretical expertise and innovation to create or apply new technology, such as adapting principles for applying computers to new uses.',
        '14627.0': 'Consult with users, management, vendors, and technicians to determine computing needs and system requirements.',
        '14634.0': 'Maintain network hardware and software, direct network security measures, and monitor networks to ensure availability to system users.',
        '14635.0': 'Participate in multidisciplinary projects in areas such as virtual reality, human-computer interaction, or robotics.',
        '14639.0': 'Test system modifications to prepare for implementation.',
        '1464.0': 'Read and review project blueprints and structural specifications to determine dimensions of structure or system and material requirements.',
        '14640.0': 'Develop testing programs that address areas such as database impacts, software scenarios, regression testing, negative testing, error or bug retests, or usability.',
        '14641.0': 'Document software defects, using a bug tracking system, and report defects to software developers.',
        '14642.0': 'Identify, analyze, and document problems with program function, output, online screen, or content.',
        '14644.0': 'Create or maintain databases of known test defects.',
        '14647.0': 'Review software documentation to ensure technical accuracy, compliance, or completeness, or to mitigate risks.',
        '14649.0': 'Develop or specify standards, methods, or procedures to determine product quality or release readiness.',
        '14652.0': 'Install, maintain, or use software testing programs.',
        '14654.0': 'Monitor program performance to ensure efficient and problem-free operations.',
        '14655.0': 'Conduct software compatibility tests with programs, hardware, operating systems, or network environments.',
        '14656.0': 'Install and configure recreations of software production environments to allow testing of software performance.',
        '14659.0': 'Design or develop automated testing tools.',
        '14661.0': 'Perform initial debugging procedures by reviewing configuration files, logs, or code pieces to determine breakdown source.',
        '14664.0': 'Conduct historical analyses of test results.',
        '14669.0': 'Verify stability, interoperability, portability, security, or scalability of system architecture.',
        '14670.0': 'Collaborate with engineers or software developers to select appropriate design solutions or ensure the compatibility of system components.',
        '14672.0': 'Provide technical guidance or support for the development or troubleshooting of systems.',
        '14673.0': 'Identify system data, hardware, or software components required to meet user needs.',
        '14674.0': 'Provide customers or installation teams guidelines for implementing secure systems.',
        '14675.0': 'Monitor system operation to detect potential problems.',
        '14676.0': 'Direct the analysis, development, and operation of complete computer systems.',
        '14678.0': 'Perform ongoing hardware and software maintenance operations, including installing or upgrading hardware or software.',
        '14679.0': 'Configure servers to meet functional specifications.',
        '14681.0': 'Define and analyze objectives, scope, issues, or organizational impact of information systems.',
        '14682.0': 'Develop system engineering, software engineering, system integration, or distributed system architectures.',
        '14683.0': 'Design and conduct hardware or software tests.',
        '14685.0': 'Evaluate existing systems to determine effectiveness, and suggest changes to meet organizational requirements.',
        '14686.0': 'Research, test, or verify proper functioning of software patches and fixes.',
        '14688.0': 'Complete models and simulations, using manual or automated tools, to analyze or predict system performance under different operating conditions.',
        '14689.0': 'Direct the installation of operating systems, network or application software, or computer or network hardware.',
        '14691.0': 'Perform security analyses of developed or packaged software components.',
        '14694.0': 'Design, build, or maintain Web sites, using authoring or scripting languages, content creation tools, management tools, and digital media.',
        '14695.0': 'Perform or direct Web site updates.',
        '14698.0': 'Back up files from Web sites to local directories for instant recovery in case of problems.',
        '14704.0': 'Develop databases that support Web applications and Web sites.',
        '14705.0': 'Renew domain name registrations.',
        '14706.0': 'Collaborate with management or users to develop e-commerce strategies and to integrate these strategies with Web sites.',
        '14707.0': 'Write supporting code for Web applications or Web sites.',
        '14708.0': 'Communicate with network personnel or Web site hosting agencies to address hardware or software issues affecting Web sites.',
        '14710.0': 'Perform Web site tests according to planned schedules, or after any Web site or product revision.',
        '14713.0': 'Respond to user email inquiries, or set up automated systems to send responses.',
        '14714.0': 'Develop or implement procedures for ongoing Web site revision.',
        '14717.0': 'Establish appropriate server directory trees.',
        '14719.0': 'Recommend and implement performance improvements.',
        '14723.0': 'Monitor security system performance logs to identify problems and notify security specialists when problems occur.',
        '14724.0': 'Create Web models or prototypes that include physical, interface, logical, or data models.',
        '14726.0': 'Document test plans, testing procedures, or test results.',
        '14728.0': 'Document technical factors such as server load, bandwidth, database performance, and browser and device types.',
        '14729.0': 'Install and configure hypertext transfer protocol (HTTP) servers and associated operating systems.',
        '14731.0': 'Back up or modify applications and related data to provide for disaster recovery.',
        '14732.0': 'Determine sources of Web page or server problems, and take action to correct such problems.',
        '14733.0': 'Review or update Web page content or links in a timely manner, using appropriate tools.',
        '14735.0': 'Implement Web site security measures, such as firewalls or message encryption.',
        '14736.0': 'Administer internet or intranet infrastructure, including Web, file, and mail servers.',
        '14737.0': 'Collaborate with development teams to discuss, analyze, or resolve usability issues.',
        '14739.0': 'Monitor Web developments through continuing education, reading, or participation in professional conferences, workshops, or groups.',
        '14740.0': 'Implement updates, upgrades, and patches in a timely manner to limit loss of service.',
        '14742.0': 'Collaborate with Web developers to create and operate internal and external Web sites, or to manage projects, such as e-marketing campaigns.',
        '14744.0': 'Gather, analyze, or document user feedback to locate or resolve sources of problems.',
        '14745.0': 'Develop Web site performance metrics.',
        '14746.0': 'Identify or address interoperability requirements.',
        '14749.0': 'Track, compile, and analyze Web site usage data.',
        '14750.0': 'Test issues such as system integration, performance, and system security on a regular schedule or after any major program modifications.',
        '14756.0': 'Perform user testing or usage analyses to determine Web sites\' effectiveness or usability.',
        '14759.0': 'Develop or document style guidelines for Web site content.',
        '14762.0': 'Set up or maintain monitoring tools on Web servers or Web sites.',
        '14956.0': 'Operate mining machines to gather coal and convey it to floors or shuttle cars.',
        '1505.0': 'Analyze organic or inorganic compounds to determine chemical or physical properties, composition, structure, relationships, or reactions, using chromatography, spectroscopy, or spectrophotometry techniques.',
        '1506.0': 'Develop, improve, or customize products, equipment, formulas, processes, or analytical methods.',
        '1508.0': 'Confer with scientists or engineers to conduct analyses of research projects, interpret test results, or develop nonstandard tests.',
        '151.0': 'Estimate quantities and cost of materials, equipment, or labor to determine project feasibility.',
        '1510.0': 'Induce changes in composition of substances by introducing heat, light, energy, or chemical catalysts for quantitative or qualitative analysis.',
        '15198.0': 'Provide users with technical support for computer problems.',
        '15205.0': 'Diagnose, troubleshoot, and resolve hardware, software, or other network and system problems, and replace defective components when necessary.',
        '15206.0': 'Configure, monitor, and maintain email applications or virus protection software.',
        '15207.0': 'Research new technologies by attending seminars, reading trade articles, or taking classes, and implement or recommend the implementation of new technologies.',
        '15208.0': 'Implement and provide technical support for voice services and equipment, such as private branch exchange, voice mail system, and telecom system.',
        '15213.0': 'Monitor and perform tests on water, food, and the environment to detect harmful microorganisms or to obtain information about sources of pollution, contamination, or infection.',
        '15215.0': 'Monitor and report incidents of infectious diseases to local and state health agencies.',
        '15216.0': 'Communicate research findings on various types of diseases to health practitioners, policy makers, and the public.',
        '15224.0': 'Analyze and interpret geological data, using computer software.',
        '15228.0': 'Maintain archive of images, photos, or previous work products.',
        '15229.0': 'Research new software or design concepts.',
        '15232.0': 'Establish or monitor quality assurance programs or activities to ensure the accuracy of laboratory results.',
        '15248.0': 'Administer active or passive manual therapeutic exercises, therapeutic massage, aquatic physical therapy, or heat, light, sound, or electrical modality treatments, such as ultrasound.',
        '15268.0': 'Appoint nominees to leadership posts, or approve such appointments.',
        '15270.0': 'Debate the merits of proposals and bill amendments during floor sessions, following the appropriate rules of procedure.',
        '15271.0': 'Develop expertise in subject matters related to committee assignments.',
        '15272.0': 'Hear testimony from constituents, representatives of interest groups, board and commission members, and others with an interest in bills or issues under consideration.',
        '15273.0': 'Keep abreast of the issues affecting constituents by making personal visits and phone calls, reading local newspapers, and viewing or listening to local broadcasts.',
        '15274.0': 'Maintain knowledge of relevant national and international current events.',
        '15277.0': 'Prepare drafts of amendments, government policies, laws, rules, regulations, budgets, programs and procedures.',
        '15280.0': 'Review bills in committee, and make recommendations about their future.',
        '15282.0': 'Serve on commissions, investigative panels, study groups, and committees in order to examine specialized areas and recommend action.',
        '15283.0': 'Vote on motions, amendments, and decisions on whether or not to report a bill out from committee to the assembly floor.',
        '15284.0': 'Write, prepare, and deliver statements for the Congressional Record.',
        '15290.0': 'Establish personal offices in local districts or states, and manage office staff.',
        '15291.0': 'Evaluate the structure, efficiency, activities, and performance of government agencies.',
        '15293.0': 'Oversee expense allowances, ensuring that accounts are balanced at the end of each fiscal year.',
        '15341.0': 'Inspect carry-on items, using x-ray viewing equipment, to determine whether items contain objects that warrant further investigation.',
        '15371.0': 'Write project proposals, grant applications, or other documents to pursue funding for environmental initiatives.',
        '15376.0': 'Research environmental sustainability issues, concerns, or stakeholder interests.',
        '1538.0': 'Compile and interpret results of tests and analyses.',
        '15418.0': 'Monitor development of new products to help identify possible problems for mass production.',
        '15544.0': 'Track attendance, participation, or performance data related to wellness events.',
        '15549.0': 'Teach fitness classes to improve strength, flexibility, cardiovascular conditioning, or general fitness of participants.',
        '15558.0': 'Maintain wellness- and fitness-related schedules, records, or reports.',
        '15562.0': 'Develop or coordinate fitness and wellness programs or services.',
        '15592.0': 'Track enrollment status of subjects and document dropout information such as dropout causes and subject contact efforts.',
        '15593.0': 'Review proposed study protocols to evaluate factors such as sample collection processes, data management plans, or potential subject risks.',
        '15594.0': 'Record adverse event and side effect data and confer with investigators regarding the reporting of events to oversight agencies.',
        '15595.0': 'Prepare study-related documentation, such as protocol worksheets, procedural manuals, adverse event reports, institutional review board documents, or progress reports.',
        '15596.0': 'Participate in the development of study protocols including guidelines for administration or data collection procedures.',
        '15608.0': 'Collaborate with investigators to prepare presentations or reports of clinical study procedures, results, and conclusions.',
        '15609.0': 'Code, evaluate, or interpret collected study data.',
        '15610.0': 'Assess eligibility of potential subjects through methods such as screening interviews, reviews of medical records, or discussions with physicians and nurses.',
        '15611.0': 'Arrange for research study sites and determine staff or equipment availability.',
        '15613.0': 'Monitor study activities to ensure compliance with protocols and with all relevant local, federal, and state regulatory and institutional polices.',
        '15614.0': 'Maintain required records of study activity including case report forms, drug dispensation records, or regulatory forms.',
        '15637.0': 'Serve as a confidential point of contact for employees to communicate with management, seek clarification on issues or dilemmas, or report irregularities.',
        '15638.0': 'Maintain documentation of compliance activities, such as complaints received or investigation outcomes.',
        '15642.0': 'Advise internal management or business partners on the implementation or operation of compliance programs.',
        '15643.0': 'Review communications such as securities sales advertising to ensure there are no violations of standards or regulations.',
        '15645.0': 'Report violations of compliance or regulatory standards to duly authorized enforcement agencies as appropriate or required.',
        '15649.0': 'Monitor compliance systems to ensure their effectiveness.',
        '15680.0': 'Select transportation routes to maximize economy by combining shipments or consolidating warehousing and distribution.',
        '15702.0': 'Participate in online forums or conferences to stay abreast of online retailing trends, techniques, or security threats.',
        '15703.0': 'Upload digital media, such as photos, video, or scanned images to online storefront, auction sites, or other shopping Web sites.',
        '15704.0': 'Order or purchase merchandise to maintain optimal inventory levels.',
        '15705.0': 'Maintain inventory of shipping supplies, such as boxes, labels, tape, bubble wrap, loose packing materials, or tape guns.',
        '15706.0': 'Integrate online retailing strategy with physical or catalogue retailing operations.',
        '15707.0': 'Determine and set product prices.',
        '15708.0': 'Disclose merchant information and terms and policies of transactions in online or offline materials.',
        '15709.0': 'Deliver e-mail confirmation of completed transactions and shipment.',
        '15710.0': 'Create, manage, or automate orders or invoices, using order management or invoicing software.',
        '15711.0': 'Create or maintain database of customer accounts.',
        '15714.0': 'Cancel orders based on customer requests or inventory or delivery problems.',
        '15716.0': 'Select and purchase technical web services, such as web hosting services, online merchant accounts, shopping cart software, payment gateway software, or spyware.',
        '15717.0': 'Promote products in online communities through weblog or discussion-forum postings, e-mail marketing programs, or online advertising.',
        '15718.0': 'Fill customer orders by packaging sold items and documentation for direct shipping or by transferring orders to manufacturers or third-party distributors.',
        '15720.0': 'Investigate sources, such as auctions, estate sales, liquidators, wholesalers, or trade shows for new items, used items, or collectibles.',
        '15721.0': 'Investigate products or markets to determine areas for opportunity or viability for merchandising specific products, using online or offline sources.',
        '15722.0': 'Initiate online auctions through auction hosting sites or auction management software.',
        '15723.0': 'Implement security practices to preserve assets, minimize liabilities, or ensure customer privacy, using parallel servers, hardware redundancy, fail-safe technology, information encryption, or firewalls.',
        '15724.0': 'Devise, select, or purchase domain name and web address.',
        '15725.0': 'Develop or revise business plans for online business, emphasizing factors such as product line, pricing, inventory, or marketing strategy.',
        '15726.0': 'Determine location for product listings to maximize exposure to online traffic.',
        '15727.0': 'Design customer interface of online storefront, using web programming or e-commerce software.',
        '15728.0': 'Correspond with online customers via electronic mail, telephone, or other electronic messaging to address questions or complaints about products, policies, or shipping methods.',
        '1573.0': 'Interpret laboratory findings or test results to identify and classify substances, materials, or other evidence collected at crime scenes.',
        '15730.0': 'Calculate purchase subtotals, taxes, and shipping costs for submission to customers.',
        '15731.0': 'Receive and process payments from customers, using electronic transaction services.',
        '15732.0': 'Prepare or organize online storefront marketing material, including product descriptions or subject lines, optimizing content to search engine criteria.',
        '15733.0': 'Purchase new or used items from online or physical sources for resale via retail or auction Web site.',
        '15734.0': 'Compose descriptions of merchandise for posting to online storefront, auction sites, or other shopping Web sites.',
        '15735.0': 'Calculate revenue, sales, and expenses, using financial accounting or spreadsheet software.',
        '1578.0': 'Identify and quantify drugs or poisons found in biological fluids or tissues, in foods, or at crime scenes.',
        '15861.0': 'Propose logistics solutions for customers.',
        '15866.0': 'Develop specifications for equipment, tools, facility layouts, or material-handling systems.',
        '15867.0': 'Review contractual commitments, customer specifications, or related information to determine logistics or support requirements.',
        '15876.0': 'Analyze or interpret logistics data involving customer service, forecasting, procurement, manufacturing, inventory, transportation, or warehousing.',
        '15895.0': 'Prepare reports on logistics performance measures.',
        '15898.0': 'Maintain databases of logistics information.',
        '15900.0': 'Develop or maintain models for logistics uses, such as cost estimating or demand forecasting.',
        '15902.0': 'Compute reporting metrics, such as on-time delivery rates, order fulfillment rates, or inventory turns.',
        '15904.0': 'Apply analytic methods or tools to understand, predict, or control logistics operations or processes.',
        '15905.0': 'Analyze logistics data, using methods such as data mining, data modeling, or cost or benefit analysis.',
        '1593.0': 'Guide clients in the development of skills or strategies for dealing with their problems.',
        '1594.0': 'Prepare and maintain all required treatment records and reports.',
        '1595.0': 'Counsel clients or patients, individually or in group sessions, to assist in overcoming dependencies, adjusting to life, or making changes.',
        '15956.0': 'Test documented disaster recovery strategies and plans.',
        '15957.0': 'Review existing disaster recovery, crisis management, or business continuity plans.',
        '1596.0': 'Collect information about clients through interviews, observation, or tests.',
        '15960.0': 'Analyze impact on, and risk to, essential business functions or information systems to identify acceptable recovery time periods and resource requirements.',
        '15962.0': 'Develop disaster recovery plans for physical locations with critical assets, such as data centers.',
        '1597.0': 'Act as client advocates to coordinate required services or to resolve emergency problems in crisis situations.',
        '15980.0': 'Monitor or track sustainability indicators, such as energy usage, natural resource usage, waste generation, and recycling.',
        '15989.0': 'Interpret results of financial analysis procedures.',
        '15990.0': 'Develop core analytical capabilities or model libraries, using advanced statistical, quantitative, or econometric techniques.',
        '15997.0': 'Apply mathematical or statistical techniques to address practical issues in finance, such as derivative valuation, securities trading, risk management, or financial market regulation.',
        '1602.0': 'Refer patients, clients, or family members to community resources or to specialists as necessary.',
        '16035.0': 'Maintain knowledge of current events and trends in such areas as money laundering and criminal tools and techniques.',
        '16037.0': 'Research or evaluate new technologies for use in fraud detection systems.',
        '16054.0': 'Create and maintain logs, records, or databases of information about fraudulent activity.',
        '16056.0': 'Conduct in-depth investigations of suspicious financial activity, such as suspected money-laundering efforts.',
        '16057.0': 'Analyze financial data to detect irregularities in areas such as billing trends, financial relationships, and regulatory compliance procedures.',
        '16067.0': 'Use computer-aided design (CAD) software to prepare or evaluate network diagrams, floor plans, or site configurations for existing facilities, renovations, or new systems.',
        '16069.0': 'Monitor and analyze system performance, such as network traffic, security, and capacity.',
        '16070.0': 'Manage user access to systems and equipment through account management and password administration.',
        '16075.0': 'Implement controls to provide security for operating systems, software, and data.',
        '1609.0': 'Gather information about community mental health needs or resources that could be used in conjunction with therapy.',
        '16099.0': 'Provide technical support to junior staff or clients.',
        '16100.0': 'Set up database clusters, backup, or recovery processes.',
        '16101.0': 'Identify, evaluate and recommend hardware or software technologies to achieve desired database performance.',
        '16102.0': 'Plan and install upgrades of database management system software to enhance database performance.',
        '16104.0': 'Identify and correct deviations from database development standards.',
        '16105.0': 'Document and communicate database schemas, using accepted notations.',
        '16106.0': 'Develop or maintain archived procedures, procedural codes, or queries for applications.',
        '16107.0': 'Develop load-balancing processes to eliminate down time for backup processes.',
        '16108.0': 'Develop data models for applications, metadata tables, views or related database structures.',
        '16109.0': 'Design databases to support business applications, ensuring system scalability, security, performance, and reliability.',
        '16110.0': 'Design database applications, such as interfaces, data transfer mechanisms, global temporary tables, data partitions, and function-based indexes to enable efficient access of the generic database structure.',
        '16111.0': 'Demonstrate database technical functionality, such as performance, security and reliability.',
        '16112.0': 'Create and enforce database development standards.',
        '16113.0': 'Collaborate with system architects, software architects, design analysts, and others to understand business or industry requirements.',
        '16115.0': 'Develop and document database architectures.',
        '16116.0': 'Test software systems or applications for software enhancements or new products.',
        '16118.0': 'Provide or coordinate troubleshooting support for data warehouses.',
        '16119.0': 'Prepare functional or technical documentation for data warehouses.',
        '16121.0': 'Verify the structure, accuracy, or quality of warehouse data.',
        '16122.0': 'Select methods, techniques, or criteria for data warehousing evaluative procedures.',
        '16123.0': 'Perform system analysis, data analysis or programming, using a variety of computer languages and procedures.',
        '16124.0': 'Map data between source systems, data warehouses, and data marts.',
        '16125.0': 'Implement business rules via stored procedures, middleware, or other technologies.',
        '16126.0': 'Develop and implement data extraction procedures from other systems, such as administration, billing, or claims.',
        '16127.0': 'Develop or maintain standards, such as organization, structure, or nomenclature, for the design of data warehouse elements, such as data architectures, models, tools, and databases.',
        '16128.0': 'Design and implement warehouse database structures.',
        '16129.0': 'Create supporting documentation, such as metadata and diagrams of entity relationships, business processes, and process flow.',
        '16131.0': 'Create or implement metadata processes and frameworks.',
        '16132.0': 'Develop data warehouse process models, including sourcing, loading, transformation, and extraction.',
        '16133.0': 'Design, implement, or operate comprehensive data warehouse systems to balance optimization of data access with batch loading and resource utilization factors, according to customer requirements.',
        '16142.0': 'Document specifications for business intelligence or information technology reports, dashboards, or other outputs.',
        '16147.0': 'Analyze technology trends to identify markets for future product development or to improve sales of existing products.',
        '16148.0': 'Analyze competitive market strategies through analysis of related product, market, or share trends.',
        '16150.0': 'Generate standard or custom reports summarizing business, financial, or economic data for review by executives, managers, clients, and other stakeholders.',
        '16152.0': 'Submit project deliverables, ensuring adherence to quality standards.',
        '16153.0': 'Monitor the performance of project team members, providing and documenting performance feedback.',
        '16155.0': 'Assess current or future customer needs and priorities by communicating directly with customers, conducting surveys, or other methods.',
        '16156.0': 'Schedule and facilitate meetings related to information technology projects.',
        '16157.0': 'Monitor or track project milestones and deliverables.',
        '16159.0': 'Initiate, review, or approve modifications to project plans.',
        '16163.0': 'Direct or coordinate activities of project personnel.',
        '16165.0': 'Coordinate recruitment or selection of project personnel.',
        '16167.0': 'Assign duties, responsibilities, and spans of authority to project personnel.',
        '16168.0': 'Prepare project status reports by collecting, analyzing, and summarizing information and trends.',
        '16169.0': 'Manage project execution to ensure adherence to budget, schedule, and scope.',
        '16170.0': 'Develop or update project plans for information technology projects including information such as project objectives, technologies, systems, information specifications, schedules, funding, and staffing.',
        '16171.0': 'Develop and manage work breakdown structure (WBS) of information technology projects.',
        '1618.0': 'Gather information about offenders\' backgrounds by talking to offenders, their families and friends, and other people who have relevant information.',
        '16194.0': 'Review or evaluate competitive products, film, music, television, and other art forms to generate new game design ideas.',
        '16195.0': 'Provide test specifications to quality assurance staff.',
        '16196.0': 'Keep abreast of game design technology and techniques, industry trends, or audience interests, reactions, and needs by reviewing current literature, talking with colleagues, participating in educational programs, attending meetings or workshops, or participating in professional organizations or conferences.',
        '16197.0': 'Create gameplay test plans for internal and external test groups.',
        '16199.0': 'Balance and adjust gameplay experiences to ensure the critical and commercial success of the product.',
        '16200.0': 'Write or supervise the writing of game text and dialogue.',
        '16201.0': 'Solicit, obtain, and integrate feedback from design and technical staff into original game design.',
        '16202.0': 'Provide feedback to production staff regarding technical game qualities or adherence to original design.',
        '16203.0': 'Prepare two-dimensional concept layouts or three-dimensional mock-ups.',
        '16204.0': 'Present new game design concepts to management and technical colleagues, including artists, animators, and programmers.',
        '16205.0': 'Prepare and revise initial game sketches using two- and three-dimensional graphical design software.',
        '16206.0': 'Oversee gameplay testing to ensure intended gaming experience and game adherence to original vision.',
        '16207.0': 'Guide design discussions between development teams.',
        '16208.0': 'Devise missions, challenges, or puzzles to be encountered in game play.',
        '16209.0': 'Develop and maintain design level documentation, including mechanics, guidelines, and mission outlines.',
        '16210.0': 'Determine supplementary virtual features, such as currency, item catalog, menu design, and audio direction.',
        '16211.0': 'Create gameplay prototypes for presentation to creative and technical staff and management.',
        '16212.0': 'Create and manage documentation, production schedules, prototyping goals, and communication plans in collaboration with production staff.',
        '16213.0': 'Consult with multiple stakeholders to define requirements and implement online features.',
        '16215.0': 'Collaborate with artists to achieve appropriate visual style.',
        '16216.0': 'Document all aspects of formal game design, using mock-up screenshots, sample menu layouts, gameplay flowcharts, and other graphical devices.',
        '16217.0': 'Create core game features, including storylines, role-play mechanics, and character biographies for a new video game or game franchise.',
        '16220.0': 'Write, review, or execute plans for testing new or established document management systems.',
        '16221.0': 'Search electronic sources, such as databases or repositories, or manual sources for information.',
        '16222.0': 'Retrieve electronic assets from repository for distribution to users, collecting and returning to repository, if necessary.',
        '16224.0': 'Prepare support documentation and training materials for end users of document management systems.',
        '16227.0': 'Implement scanning or other automated data entry procedures, using imaging devices and document imaging software.',
        '16228.0': 'Document technical functions and specifications for new or proposed content management systems.',
        '16229.0': 'Develop, document, or maintain standards, best practices, or system usage procedures.',
        '16230.0': 'Consult with end users regarding problems in accessing electronic content.',
        '16231.0': 'Conduct needs assessments to identify document management requirements of departments or end users.',
        '16232.0': 'Assist in the development of document or content classification taxonomies to facilitate information capture, search, and retrieval.',
        '16233.0': 'Assist in the assessment, acquisition, or deployment of new electronic document management systems.',
        '16236.0': 'Operate data capture technology to import digitized documents into document management system.',
        '16237.0': 'Administer document and system access rights and revision control to ensure security of system and integrity of master documents.',
        '16238.0': 'Implement electronic document processing, retrieval, and distribution systems in collaboration with other information technology specialists.',
        '16239.0': 'Identify and classify documents or other electronic content according to characteristics such as security level, function, and metadata.',
        '16240.0': 'Develop or configure document management system features, such as user interfaces, access profiles, and document workflow procedures.',
        '16246.0': 'Calculate sample size requirements for clinical studies.',
        '16249.0': 'Write program code to analyze data with statistical analysis software.',
        '16252.0': 'Prepare tables and graphs to present clinical data or results.',
        '16254.0': 'Draw conclusions or make predictions, based on data summaries or statistical analyses.',
        '16255.0': 'Develop or use mathematical models to track changes in biological phenomena, such as the spread of infectious diseases.',
        '16257.0': 'Develop or implement data analysis algorithms.',
        '16259.0': 'Design or maintain databases of biological data.',
        '16262.0': 'Review clinical or other medical research protocols and recommend appropriate statistical analyses.',
        '16263.0': 'Provide biostatistical consultation to clients or colleagues.',
        '16264.0': 'Apply research or simulation results to extend biological theory or recommend new research projects.',
        '16265.0': 'Analyze clinical or survey data, using statistical approaches such as longitudinal analysis, mixed-effect modeling, logistic regression analyses, and model-building techniques.',
        '16277.0': 'Prepare data analysis listings and activity, performance, or progress reports.',
        '16278.0': 'Perform quality control audits to ensure accuracy, completeness, or proper usage of clinical systems and data.',
        '16323.0': 'Design or prepare plans for new transportation systems or parts of systems, such as airports, commuter trains, highways, streets, bridges, drainage structures, or roadway lighting.',
        '16332.0': 'Perform mathematical modeling of underground or surface water resources, such as floodplains, ocean coastlines, streams, rivers, or wetlands.',
        '16334.0': 'Perform hydraulic analyses of water supply systems or water distribution networks to model flow characteristics, test for pressure losses, or to identify opportunities to mitigate risks and improve operational efficiency.',
        '16378.0': 'Conduct interviews or surveys of users or customers to collect information on topics, such as requirements, needs, fatigue, ergonomics, or interfaces.',
        '16434.0': 'Calibrate vehicle systems, including control algorithms or other software systems.',
        '1644.0': 'Develop and maintain an institution\'s registration, cataloging, and basic record-keeping systems, using computer databases.',
        '1645.0': 'Provide information from the institution\'s holdings to other curators and to the public.',
        '1646.0': 'Inspect premises to assess the need for repairs and to ensure that climate and pest control issues are addressed.',
        '16467.0': 'Create mechanical design documents for parts, assemblies, or finished products.',
        '16470.0': 'Implement or test design solutions.',
        '1649.0': 'Plan and conduct special research projects in area of interest or expertise.',
        '16511.0': 'Provide technical support for robotic systems.',
        '16514.0': 'Document robotic application development, maintenance, or changes.',
        '16515.0': 'Write algorithms or programming code for ad hoc robotic applications.',
        '16520.0': 'Install, calibrate, operate, or maintain robots.',
        '16528.0': 'Design software to control robotic systems for applications, such as military defense or manufacturing.',
        '16569.0': 'Perform computer simulation of solar photovoltaic (PV) generation system performance or energy production to optimize efficiency.',
        '16582.0': 'Develop three-dimensional simulations of automation systems.',
        '16584.0': 'Modify computer-controlled robot movements.',
        '16585.0': 'Develop robotic path motions to maximize efficiency, safety, and quality.',
        '16618.0': 'Make radiographic images to detect flaws in objects while leaving objects intact.',
        '16779.0': 'Collaborate with software developers in the development and modification of commercial bioinformatics software.',
        '16780.0': 'Test new and updated bioinformatics tools and software.',
        '16781.0': 'Provide statistical and computational tools for biologically based activities, such as genetic analysis, measurement of gene expression, or gene function determination.',
        '16782.0': 'Prepare summary statistics of information regarding human genomes.',
        '16783.0': 'Instruct others in the selection and use of bioinformatics tools.',
        '16784.0': 'Improve user interfaces to bioinformatics software and databases.',
        '16786.0': 'Develop new software applications or customize existing applications to meet specific scientific project needs.',
        '16788.0': 'Create or modify web-based bioinformatics tools.',
        '16789.0': 'Design and apply bioinformatics algorithms including unsupervised and supervised machine learning, dynamic programming, or graphic algorithms.',
        '16790.0': 'Create novel computational approaches and analytical tools as required by research goals.',
        '16791.0': 'Compile data for use in activities, such as gene expression profiling, genome annotation, or structural bioinformatics.',
        '16792.0': 'Communicate research results through conference presentations, scientific publications, or project reports.',
        '16793.0': 'Manipulate publicly accessible, commercial, or proprietary genomic, proteomic, or post-genomic databases.',
        '16794.0': 'Consult with researchers to analyze problems, recommend technology-based solutions, or determine computational strategies.',
        '16795.0': 'Analyze large molecular datasets, such as raw microarray data, genomic sequence data, or proteomics data, for clinical or basic research purposes.',
        '16797.0': 'Participate in all levels of bioproduct development, including proposing new products, performing market analyses, designing and performing experiments, and collaborating with operations and quality control teams during product launches.',
        '16801.0': 'Confer with vendors to evaluate new equipment or reagents or to discuss the customization of product lines to meet user requirements.',
        '16806.0': 'Monitor or operate specialized equipment, such as gas chromatographs and high pressure liquid chromatographs, electrophoresis units, thermocyclers, fluorescence activated cell sorters, and phosphorimagers.',
        '16807.0': 'Maintain accurate laboratory records and data.',
        '16809.0': 'Evaluate new technologies to enhance or complement current research.',
        '16811.0': 'Develop assays that monitor cell characteristics.',
        '16813.0': 'Design molecular or cellular laboratory experiments, oversee their execution, and interpret results.',
        '16815.0': 'Conduct research on cell organization and function, including mechanisms of gene expression, cellular bioinformatics, cell signaling, or cell differentiation.',
        '16816.0': 'Conduct applied research aimed at improvements in areas such as disease testing, crop quality, pharmaceuticals, and the harnessing of microbes to recycle waste.',
        '16824.0': 'Design and maintain genetics computer databases.',
        '16829.0': 'Review, approve, or interpret genetic laboratory results.',
        '16839.0': 'Plan or conduct basic genomic and biological research related to areas such as regulation of gene expression, protein interactions, metabolic networks, and nucleic acid or protein complexes.',
        '16840.0': 'Analyze determinants responsible for specific inherited traits, and devise methods for altering traits or producing new traits.',
        '16846.0': 'Gather and review climate-related studies from government agencies, research laboratories, and other organizations.',
        '1688.0': 'Reserve, circulate, renew, and discharge books and other materials.',
        '16884.0': 'Plan or conduct studies of the ecological implications of historic or projected changes in industrial processes or development.',
        '16886.0': 'Monitor the environmental impact of development activities, pollution, or land degradation.',
        '1695.0': 'Conduct reference searches, using printed materials and in-house and online databases.',
        '16954.0': 'Analyze information related to transportation, such as land use policies, environmental impact of projects, or long-range planning needs.',
        '16984.0': 'Develop specialized computer software routines to customize and integrate image analysis.',
        '16985.0': 'Collect verification data on the ground, using equipment such as global positioning receivers, digital cameras, or notebook computers.',
        '16986.0': 'Verify integrity and accuracy of data contained in remote sensing image analysis systems.',
        '16987.0': 'Prepare documentation or presentations, including charts, photos, or graphs.',
        '16990.0': 'Merge scanned images or build photo mosaics of large areas, using image processing software.',
        '16991.0': 'Integrate remotely sensed data with other geospatial data.',
        '16992.0': 'Evaluate remote sensing project requirements to determine the types of equipment or computer software necessary to meet project requirements, such as specific image types or output resolutions.',
        '16993.0': 'Develop or maintain geospatial information databases.',
        '16994.0': 'Correct raw data for errors due to factors such as skew or atmospheric variation.',
        '16997.0': 'Adjust remotely sensed images for optimum presentation by using software to select image displays, define image set categories, or choose processing routines.',
        '16999.0': 'Collect geospatial data, using technologies such as aerial photography, light and radio wave detection systems, digital satellites, or thermal energy systems.',
        '1700.0': 'Retrieve information from central databases for storage in a library\'s computer.',
        '17020.0': 'Schedule tutoring appointments with students or their parents.',
        '17029.0': 'Maintain records of students\' assessment results, progress, feedback, or school performance, ensuring confidentiality of all records.',
        '17031.0': 'Develop teaching or training materials, such as handouts, study materials, or quizzes.',
        '17034.0': 'Assess students\' progress throughout tutoring sessions.',
        '17037.0': 'Provide private instruction to individual or small groups of students to improve academic performance, improve occupational skills, or prepare for academic or occupational tests.',
        '17068.0': 'Order or perform diagnostic tests such as skin pricks and intradermal, patch, or delayed hypersensitivity tests.',
        '17076.0': 'Diagnose or treat allergic or immunologic conditions.',
        '17078.0': 'Record patients\' health histories.',
        '17095.0': 'Diagnose and treat skin conditions such as acne, dandruff, athlete\'s foot, moles, psoriasis, or skin cancer.',
        '1710.0': 'Operate and maintain audio-visual equipment, such as projectors, tape recorders, and videocassette recorders.',
        '17121.0': 'Order or interpret results of laboratory analyses of patients\' blood or cerebrospinal fluid.',
        '17170.0': 'Perform, order, or interpret the results of diagnostic or clinical tests.',
        '17181.0': 'Develop or adopt new tests or instruments to improve diagnosis of diseases.',
        '17191.0': 'Analyze and interpret results from tests, such as microbial or parasite tests, urine analyses, hormonal assays, fine needle aspirations (FNAs), and polymerase chain reactions (PCRs).',
        '17193.0': 'Consult with physicians about ordering and interpreting tests or providing treatments.',
        '17194.0': 'Examine microscopic samples to identify diseases or other abnormalities.',
        '17201.0': 'Document examination results, treatment plans, and patients\' outcomes.',
        '17218.0': 'Design or use surveillance tools, such as screening, lab reports, and vital records, to identify health risks.',
        '17252.0': 'Communicate examination results or diagnostic information to referring physicians, patients, or families.',
        '17258.0': 'Observe and evaluate athletes\' mental well-being.',
        '17260.0': 'Develop and prescribe exercise programs, such as off-season conditioning regimens.',
        '17262.0': 'Conduct research in the prevention or treatment of injuries or medical conditions related to sports and exercise.',
        '17264.0': 'Attend games and competitions to provide evaluation and treatment of activity-related injuries or medical conditions.',
        '17265.0': 'Record athletes\' medical histories, and perform physical examinations.',
        '17266.0': 'Supervise the rehabilitation of injured athletes.',
        '17268.0': 'Record athletes\' medical care information, and maintain medical records.',
        '17270.0': 'Provide education and counseling on illness and injury prevention.',
        '17274.0': 'Inform athletes about nutrition, hydration, dietary supplements, or uses and possible consequences of medication.',
        '17294.0': 'Examine patients using equipment, such as radiograph (x-ray) machines or fluoroscopes, to determine the nature and extent of disorder or injury.',
        '17298.0': 'Collect samples or specimens for diagnostic testing.',
        '17356.0': 'Formulate herbal preparations to treat conditions considering herbal properties, such as taste, toxicity, effects of preparation, contraindications, and incompatibilities.',
        '17364.0': 'Evaluate treatment outcomes and recommend new or altered treatments as necessary to further promote, restore, or maintain health.',
        '17369.0': 'Analyze physical findings and medical histories to make diagnoses according to Oriental medicine traditions.',
        '17370.0': 'Develop individual treatment plans and strategies.',
        '17426.0': 'Calibrate, troubleshoot, or repair equipment and correct malfunctions, as needed.',
        '17472.0': 'Conduct ongoing prenatal health assessments, tracking changes in physical and emotional health.',
        '17475.0': 'Estimate patients\' due dates and re-evaluate as necessary based on examination results.',
        '17482.0': 'Maintain documentation of all patients\' contacts, reviewing and updating records as necessary.',
        '175.0': 'Collect and analyze biological data about relationships among and between organisms and their environment.',
        '17534.0': 'Collect and compile data to document clients\' performance or assess program quality.',
        '17543.0': 'Study activities relating to narcotics, money laundering, gangs, auto theft rings, terrorism, or other national security threats.',
        '17546.0': 'Evaluate records of communications, such as telephone calls, to plot activity and determine the size and location of criminal groups and members.',
        '17551.0': 'Operate cameras, radios, or other surveillance equipment to intercept communications or document activities.',
        '17554.0': 'Study the assets of criminal suspects to determine the flow of money from or to targeted groups.',
        '17587.0': 'Provide customers with product details, such as coffee blend or preparation descriptions.',
        '17588.0': 'Receive and process customer payments.',
        '17610.0': 'Perform accounting duties, such as recording daily cash flow, preparing bank deposits, or generating financial statements.',
        '17667.0': 'Explain policies, procedures, or services to patients using medical or administrative knowledge.',
        '17671.0': 'Refer patients to appropriate health care services or resources.',
        '17680.0': 'Select shipment routes, based on nature of goods shipped, transit times, or security needs.',
        '17681.0': 'Determine efficient and cost-effective methods of moving goods from one location to another.',
        '17682.0': 'Reserve necessary space on ships, aircraft, trains, or trucks.',
        '17683.0': 'Arrange delivery or storage of goods at destinations.',
        '17687.0': 'Prepare shipping documentation, such as bills of lading, packing lists, dock receipts, or certificates of origin.',
        '17689.0': 'Inform clients of factors such as shipping options, timelines, transfers, or regulations affecting shipments.',
        '17690.0': 'Keep records of goods dispatched or received.',
        '17692.0': 'Monitor or record locations of goods in transit.',
        '17694.0': 'Obtain or arrange cargo insurance.',
        '17696.0': 'Prepare invoices or cost quotations for freight transportation.',
        '17707.0': 'Enter or retrieve information from structural databases, protein sequence motif databases, mutation databases, genomic databases or gene expression databases.',
        '17709.0': 'Analyze or manipulate bioinformatics data using software packages, statistical applications, or data mining techniques.',
        '17712.0': 'Design or implement web-based tools for querying large-scale biological databases.',
        '17721.0': 'Conduct quality analyses of data inputs and resulting analyses or predictions.',
        '17723.0': 'Package bioinformatics data for submission to public repositories.',
        '17758.0': 'Measure and analyze system performance and operating parameters to assess operating condition of systems or equipment.',
        '1777.0': 'Read copy or proof to detect and correct errors in spelling, punctuation, and syntax.',
        '178.0': 'Identify, classify, and study structure, behavior, ecology, physiology, nutrition, culture, and distribution of plant and animal species.',
        '1780.0': 'Verify facts, dates, and statistics, using standard reference sources.',
        '1782.0': 'Develop story or content ideas, considering reader or audience appeal.',
        '1786.0': 'Read, evaluate and edit manuscripts or other materials submitted for publication, and confer with authors regarding changes in content, style or organization, or publication.',
        '17872.0': 'Sort materials, such as metals, glass, wood, paper or plastics, into appropriate containers for recycling.',
        '18022.0': 'Direct the preparation and submission of regulatory agency applications, reports, or correspondence.',
        '18024.0': 'Provide regulatory guidance to departments or development project teams regarding design, development, evaluation, or marketing of products.',
        '18029.0': 'Investigate product complaints and prepare documentation and submissions to appropriate regulatory agencies as necessary.',
        '18030.0': 'Maintain current knowledge of relevant regulations, including proposed and final rules.',
        '18031.0': 'Manage activities such as audits, regulatory agency inspections, or product recalls.',
        '18032.0': 'Monitor emerging trends regarding industry regulations to determine potential impacts on organizational processes.',
        '18035.0': 'Provide responses to regulatory agencies regarding product information or issues.',
        '18036.0': 'Represent organizations before domestic or international regulatory agencies on major policy matters or decisions regarding company products.',
        '18037.0': 'Review all regulatory agency submission materials to ensure timeliness, accuracy, comprehensiveness, or compliance with regulatory standards.',
        '18038.0': 'Review materials such as marketing literature or user manuals to ensure that regulatory agency requirements are met.',
        '18045.0': 'Communicate with regulatory agencies regarding pre-submission strategies, potential regulatory pathways, compliance test requirements, or clarification and follow-up of submissions under review.',
        '18048.0': 'Coordinate, prepare, or review regulatory submissions for domestic or international projects.',
        '18053.0': 'Compile and maintain regulatory documentation databases or systems.',
        '18054.0': 'Coordinate efforts associated with the preparation of regulatory documents or submissions.',
        '18058.0': 'Maintain current knowledge base of existing and emerging regulations, standards, or guidance documents.',
        '18059.0': 'Obtain and distribute updated information regarding domestic or international laws, guidelines, or standards.',
        '18075.0': 'Analyze energy bills, including utility rates or tariffs, to gather historical energy usage data.',
        '18077.0': 'Calculate potential for energy savings.',
        '18078.0': 'Collect and analyze field data related to energy usage.',
        '18088.0': 'Design, develop, select, test, implement, and evaluate new or modified informatics solutions, data structures, and decision-support mechanisms to support patients, health care professionals, and their information management and human-computer and human-technology interactions within health care contexts.',
        '18180.0': 'Review product designs for manufacturability or completeness.',
        '18256.0': 'Analyze data acquired from aircraft, satellites, or ground-based platforms, using statistical analysis software, image analysis software, or Geographic Information Systems (GIS).',
        '18257.0': 'Develop or build databases for remote sensing or related geospatial project information.',
        '18258.0': 'Integrate other geospatial data sources into projects.',
        '18260.0': 'Organize and maintain geospatial data and associated documentation.',
        '18261.0': 'Process aerial or satellite imagery to create products such as land cover maps.',
        '18263.0': 'Direct all activity associated with implementation, operation, or enhancement of remote sensing hardware or software.',
        '18265.0': 'Compile and format image data to increase its usefulness.',
        '18270.0': 'Manage or analyze data obtained from remote sensing systems to obtain meaningful results.',
        '18276.0': 'Set up or maintain remote sensing data collection systems.',
        '18282.0': 'Identify spatial coordinates, using remote sensing and Global Positioning System (GPS) data.',
        '18289.0': 'Create, layer, and analyze maps showing precision agricultural data, such as crop yields, soil characteristics, input applications, terrain, drainage patterns, or field management history.',
        '18308.0': 'Document data related to patients\' care, including assessment results, interventions, medications, patient responses, or treatment changes.',
        '18311.0': 'Obtain specimens or samples for laboratory work.',
        '18316.0': 'Administer blood and blood product transfusions or intravenous infusions, monitoring patients for adverse reactions.',
        '18327.0': 'Evaluate patients\' vital signs or laboratory data to determine emergency intervention needs.',
        '18335.0': 'Assess patients\' pain levels or sedation requirements.',
        '18381.0': 'Order, perform, or interpret the results of diagnostic tests, such as complete blood counts (CBCs), electrocardiograms (EKGs), and radiographs (x-rays).',
        '18382.0': 'Analyze and interpret patients\' histories, symptoms, physical findings, or diagnostic information to develop appropriate diagnoses.',
        '18391.0': 'Perform routine or annual physical examinations.',
        '18399.0': 'Maintain complete and detailed records of patients\' health care plans and prognoses.',
        '1841.0': 'Record patients\' medical information and vital signs.',
        '18412.0': 'Educate patients about health care management.',
        '18413.0': 'Advise patients about therapeutic exercise and nutritional medicine regimens.',
        '18422.0': 'Monitor updates from public health agencies to keep abreast of health trends.',
        '18423.0': 'Conduct periodic public health maintenance activities such as immunizations and screenings for diseases and disease risk factors.',
        '18424.0': 'Obtain medical records from previous physicians or other health care providers for the purpose of patient evaluation.',
        '18438.0': 'Present or publish scientific papers.',
        '18456.0': 'Write information in medical records or provide narrative summaries to communicate patient information to other health care providers.',
        '18472.0': 'Evaluate chemical equipment and processes to identify ways to optimize performance or to ensure compliance with safety and environmental regulations.',
        '18492.0': 'Prepare for trial by performing tasks such as organizing exhibits.',
        '18510.0': 'Help patrons find and use library resources, such as reference materials, audio-visual equipment, computers, and other electronic resources and provide technical assistance when needed.',
        '18518.0': 'Compile data and create statistical reports on library usage.',
        '18520.0': 'Review and detail shop drawings for construction plans.',
        '18521.0': 'Use computer-aided drafting (CAD) and related software to produce construction documents.',
        '18559.0': 'Answer customer questions regarding problems with their accounts.',
        '18561.0': 'Prepare and process payroll information.',
        '18562.0': 'Process paperwork for new employees and enter employee information into the payroll system.',
        '18565.0': 'Confer with customers by telephone or in person to provide information about products or services, take or enter orders, cancel accounts, or obtain details of complaints.',
        '18567.0': 'Answer inquiries pertaining to hotel services, guest registration, and travel directions, or make recommendations regarding shopping, dining, or entertainment.',
        '18572.0': 'Process, verify, and maintain personnel related documentation, including staffing, recruitment, training, grievances, performance evaluations, classifications, and employee leaves of absence.',
        '18593.0': 'Advise students on issues such as course selection, progress toward graduation, and career decisions.',
        '18599.0': 'Monitor the revenue activity of the hotel or facility.',
        '18603.0': 'Conduct training or in-services to educate clinicians and other personnel on proper use of equipment.',
        '18611.0': 'Develop new food items for production, based on consumer feedback.',
        '18612.0': 'Stay up to date on new regulations and current events regarding food science by reviewing scientific literature.',
        '18633.0': 'Establish contacts with employers to create internship and employment opportunities for students.',
        '18634.0': 'Plan, direct, and participate in recruitment and enrollment activities.',
        '18638.0': 'Assist students who need extra help, such as by tutoring and preparing and implementing remedial programs.',
        '18641.0': 'Plan and organize the acquisition, storage, and exhibition of collections and related materials, including the selection of exhibition themes and designs, and develop or install exhibit materials.',
        '18642.0': 'Design, organize, or conduct tours, workshops, and instructional or educational sessions to acquaint individuals with an institution\'s facilities and materials.',
        '18645.0': 'Lead tours and teach educational courses to students and the general public.',
        '18646.0': 'Deliver artwork on courier trips.',
        '18649.0': 'Install, adjust, and operate electronic equipment to record, edit, and transmit radio and television programs, motion pictures, video conferencing, or multimedia presentations.',
        '18650.0': 'Diagnose and resolve media system problems.',
        '18651.0': 'Reserve audio-visual equipment and facilities, such as meeting rooms.',
        '18652.0': 'Analyze and maintain data logs for audio-visual activities.',
        '18654.0': 'Play and record broadcast programs, using automation systems.',
        '18655.0': 'Set up, operate, and maintain broadcast station computers and networks.',
        '18661.0': 'Convert video and audio recordings into digital formats for editing or archiving.',
        '18662.0': 'Set up, test, and adjust recording equipment for recording sessions and live performances.',
        '1869.0': 'Administer hearing or speech and language evaluations, tests, or examinations to patients to collect information on type and degree of impairments, using written or oral tests or special instruments.',
        '18698.0': 'Complete administrative tasks, such as entering orders into computer, answering telephone calls, or maintaining medical or facility information.',
        '18708.0': 'Maintain records of evidence and write and review reports.',
        '18746.0': 'Assist host or hostess by answering phones to take reservations or to-go orders, and by greeting, seating, and thanking guests.',
        '18748.0': 'Provide guests with information about local areas, including directions.',
        '18765.0': 'Assist customers by providing information and resolving their complaints.',
        '18794.0': 'Drive vehicles over specified routes or to specified destinations according to time schedules, complying with traffic regulations to ensure that passengers have a smooth and safe ride.',
        '18853.0': 'Analyze employment-related data and prepare required reports.',
        '18854.0': 'Conduct exit interviews and ensure that necessary employment termination paperwork is completed.',
        '18855.0': 'Conduct reference or background checks on job applicants.',
        '18857.0': 'Contact job applicants to inform them of the status of their applications.',
        '18858.0': 'Develop or implement recruiting strategies to meet current or anticipated staffing needs.',
        '18859.0': 'Hire employees and process hiring-related paperwork.',
        '18861.0': 'Interpret and explain human resources policies, procedures, laws, standards, or regulations.',
        '18863.0': 'Maintain and update human resources documents, such as organizational charts, employee handbooks or directories, or performance evaluation forms.',
        '18865.0': 'Perform searches for qualified job candidates, using sources such as computer databases, networking, Internet recruiting resources, media advertisements, job fairs, recruiting firms, or employee referrals.',
        '18866.0': 'Prepare or maintain employment records related to events, such as hiring, termination, leaves, transfers, or promotions, using human resources management system software.',
        '18868.0': 'Review employment applications and job orders to match applicants with job requirements.',
        '1889.0': 'Enter data from analysis of medical tests or clinical results into computer for storage.',
        '1890.0': 'Analyze samples of biological material for chemical content or reaction.',
        '18933.0': 'Advise clients or respond to inquiries about financial matters in person or via phone, email, Web site, or Internet chat.',
        '18934.0': 'Assess clients\' overall financial situations by reviewing income, assets, debts, expenses, credit reports, or other financial information.',
        '18936.0': 'Create debt management plans, spending plans, or budgets to assist clients to meet financial goals.',
        '18937.0': 'Estimate time for debt repayment, given amount of debt, interest rates, and available funds.',
        '18939.0': 'Interview clients by telephone or in person to gather financial information.',
        '18944.0': 'Recommend educational materials or resources to clients on matters, such as financial planning, budgeting, or credit.',
        '1895.0': 'Develop, standardize, evaluate, or modify procedures, techniques, or tests used in the analysis of specimens or in medical laboratory experiments.',
        '18956.0': 'Adjust network sizes to meet volume or capacity demands.',
        '18957.0': 'Communicate with customers, sales staff, or marketing staff to determine customer needs.',
        '18958.0': 'Communicate with system users to ensure accounts are set up properly or to diagnose and solve operational problems.',
        '18959.0': 'Coordinate installation of new equipment.',
        '18960.0': 'Coordinate network operations, maintenance, repairs, or upgrades.',
        '18961.0': 'Coordinate network or design activities with designers of associated networks.',
        '18962.0': 'Design, build, or operate equipment configuration prototypes, including network hardware, software, servers, or server operation systems.',
        '18964.0': 'Determine specific network hardware or software requirements, such as platforms, interfaces, bandwidths, or routine schemas.',
        '18965.0': 'Develop and implement solutions for network problems.',
        '18967.0': 'Develop conceptual, logical, or physical network designs.',
        '18969.0': 'Develop network-related documentation.',
        '18971.0': 'Develop or recommend network security measures, such as firewalls, network security audits, or automated security probes.',
        '18973.0': 'Develop procedures to track, project, or report network availability, reliability, capacity, or utilization.',
        '18977.0': 'Maintain networks by performing activities such as file addition, deletion, or backup.',
        '18979.0': 'Monitor and analyze network performance and reports on data input or output to detect problems, identify inefficient use of computer resources, or perform capacity planning.',
        '18982.0': 'Prepare detailed network specifications, including diagrams, charts, equipment configurations, or recommended technologies.',
        '18984.0': 'Research and test new or modified hardware or software products to determine performance and interoperability.',
        '18988.0': 'Communicate with vendors to gather information about products, alert them to future needs, resolve problems, or address system maintenance issues.',
        '18989.0': 'Analyze network data to determine network usage, disk space availability, or server function.',
        '18990.0': 'Configure and define parameters for installation or testing of local area network (LAN), wide area network (WAN), hubs, routers, switches, controllers, multiplexers, or related networking equipment.',
        '18991.0': 'Configure security settings or access permissions for groups or individuals.',
        '18992.0': 'Configure wide area network (WAN) or local area network (LAN) routers or related equipment.',
        '18993.0': 'Document network support activities.',
        '18994.0': 'Evaluate local area network (LAN) or wide area network (WAN) performance data to ensure sufficient availability or speed, to identify network problems, or for disaster recovery purposes.',
        '18995.0': 'Identify the causes of networking problems, using diagnostic testing software and equipment.',
        '18996.0': 'Install and configure wireless networking equipment.',
        '18997.0': 'Install network software, including security or firewall software.',
        '18998.0': 'Install new hardware or software systems or components, ensuring integration with existing network systems.',
        '18999.0': 'Install or repair network cables, including fiber optic cables.',
        '19000.0': 'Perform routine maintenance or standard repairs to networking components or equipment.',
        '19002.0': 'Troubleshoot network or connectivity problems for users or user groups.',
        '19004.0': 'Back up network data.',
        '19008.0': 'Maintain logs of network activity.',
        '19009.0': 'Monitor industry Web sites or publications for information about patches, releases, viruses, or potential problem identification.',
        '19010.0': 'Provide telephone support related to networking or connectivity issues.',
        '19011.0': 'Research hardware or software products to meet technical networking or security needs.',
        '19017.0': 'Advise clients or community groups on issues related to improving general health, such as diet or exercise.',
        '19020.0': 'Advise clients or community groups on issues related to self-care, such as diabetes management.',
        '19029.0': 'Identify the particular health care needs of individuals in a community or target area.',
        '19040.0': 'Collect information from individuals to compile vital statistics about the general health of community members.',
        '19044.0': 'Attend court sessions to hear oral arguments or record necessary case information.',
        '19045.0': 'Communicate with counsel regarding case management or procedural requirements.',
        '19051.0': 'Research laws, court decisions, documents, opinions, briefs, or other information related to cases before the court.',
        '19053.0': 'Review dockets of pending litigation to ensure adequate progress.',
        '19057.0': 'Enter information into computerized court calendar, filing, or case management systems.',
        '19097.0': 'Prepare reports on students and activities as required by administration.',
        '19171.0': 'Design music therapy experiences, using various musical elements to meet client\'s goals or objectives.',
        '19176.0': 'Observe and document client reactions, progress, or other outcomes related to music therapy.',
        '19186.0': 'Conduct, or assist in the conduct of, music therapy research.',
        '19225.0': 'Operate magnetic resonance imaging (MRI) scanners.',
        '19331.0': 'Provide physical support to assist patients to perform daily living activities, such as getting out of bed, bathing, dressing, using the toilet, standing, walking, or exercising.',
        '19348.0': 'Transport specimens, laboratory items, or pharmacy items, ensuring proper documentation and delivery to authorized personnel.',
        '19384.0': 'Calibrate or maintain machines, such as those used for plasma collection.',
        '19392.0': 'Process blood or other fluid samples for further analysis by other medical professionals.',
        '19471.0': 'Load digital images onto computers directly from cameras or from storage devices, such as flash memory cards or universal serial bus (USB) devices.',
        '19476.0': 'Operate scanners or related computer equipment to digitize negatives, photographic prints, or other images.',
        '19477.0': 'Operate special equipment to perform tasks such as transferring film to videotape or producing photographic enlargements.',
        '19479.0': 'Produce color or black-and-white photographs, negatives, or slides, applying standard photographic reproduction techniques and procedures.',
        '19482.0': 'Retouch photographic negatives or original prints to correct defects.',
        '19755.0': 'Analyze climate data sets, using techniques such as geophysical fluid dynamics, data assimilation, or numerical modeling.',
        '19756.0': 'Conduct wind assessment, integration, or validation studies.',
        '19758.0': 'Estimate or predict the effects of global warming over time for specific geographic regions.',
        '19759.0': 'Formulate predictions by interpreting environmental data, such as meteorological, atmospheric, oceanic, paleoclimate, climate, or related information.',
        '19806.0': 'Provide remote sensing data for use in addressing environmental issues, such as surface water modeling or dust cloud detection.',
        '19817.0': 'Test or balance newly installed HVAC systems to determine whether indoor air quality standards are met.',
        '19829.0': 'Analyze shipping routes to determine how to minimize environmental impact.',
        '19834.0': 'Compare shipping routes or methods to determine which have the least environmental impact.',
        '19860.0': 'Perform building commissioning activities by completing mechanical inspections of a building\'s water, lighting, or heating, ventilating, and air conditioning (HVAC) systems.',
        '19862.0': 'Verify that heating, ventilating, and air conditioning (HVAC) systems are designed, installed, and calibrated in accordance with green certification standards, such as those of Leadership in Energy and Environmental Design (LEED).',
        '199.0': 'Plan and supervise forestry projects, such as determining the type, number and placement of trees to be planted, managing tree nurseries, thinning forest and monitoring growth of new seedlings.',
        '19923.0': 'Plan or adjust routes based on changing conditions, using computer equipment, global positioning systems (GPS) equipment, or other navigation devices, to minimize fuel consumption and carbon emissions.',
        '19972.0': 'Identify environmental impacts caused by products, systems, or projects.',
        '20009.0': 'Remotely monitor the flow of vehicles or inventory, using Web-based logistics information systems to track vehicles or containers.',
        '20061.0': 'Prepare materials for laboratory activities and course materials, such as syllabi, homework assignments, and handouts.',
        '20065.0': 'Participate in campus and community events, such as giving presentations to the public.',
        '20067.0': 'Provide information to the public by leading workshops and training programs and by developing educational materials.',
        '20069.0': 'Review papers or serve on editorial boards for scientific journals, and review grant proposals for federal agencies.',
        '20080.0': 'Develop and use multimedia course materials and other current technology, such as online courses.',
        '20138.0': 'Supervise, train, and evaluate residence hall staff, including resident assistants, participants in work-study programs, and other student workers.',
        '20141.0': 'Process loan applications.',
        '20168.0': 'Monitor financial activities and details, such as cash flow and reserve levels, to ensure that all legal and regulatory requirements are met.',
        '20172.0': 'Prepare budgets, bids, or contracts.',
        '20200.0': 'Conduct research to develop methodologies, instrumentation, and procedures for medical application, analyzing data and presenting findings to the scientific audience and general public.',
        '20205.0': 'Analyze historical climate information, such as precipitation or temperature records, to help predict future weather or climate trends.',
        '20206.0': 'Prepare weather reports or maps for analysis, distribution, or use in weather broadcasts, using computer graphics.',
        '20207.0': 'Apply meteorological knowledge to issues such as global warming, pollution control, or ozone depletion.',
        '20208.0': 'Develop or use mathematical or computer models for weather forecasting.',
        '20209.0': 'Interpret data, reports, maps, photographs, or charts to predict long- or short-range weather conditions, using computer models and knowledge of climate theory, physics, and mathematics.',
        '20212.0': 'Research the impact of industrial projects or pollution on climate, air quality, or weather phenomena.',
        '20213.0': 'Perform experiments and computer modeling to study the nature, structure, and physical and chemical properties of metals and their alloys, and their responses to applied forces.',
        '20226.0': 'Write articles, white papers, or reports to share research findings and educate others.',
        '20239.0': 'Conduct hearings to obtain information or evidence relative to disposition of claims.',
        '20243.0': 'Develop or maintain online help documentation.',
        '20267.0': 'Prepare sales contracts and order forms.',
        '20268.0': 'Negotiate details of contracts and payments.',
        '20271.0': 'Track delivery progress of shipments.',
        '20279.0': 'Determine shipping methods, routes, or rates for materials to be shipped.',
        '20286.0': 'Collect and deposit money into accounts, disburse funds from cash accounts to pay bills or invoices, keep records of collections and disbursements, and ensure accounts are balanced.',
        '20288.0': 'Develop or maintain internal or external company Web sites.',
        '20332.0': 'Plan, schedule, or coordinate construction project activities to meet deadlines.',
        '2034.0': 'Schedule appointments for patients.',
        '2036.0': 'Greet and log in patients arriving at office or clinic.',
        '2037.0': 'Contact medical facilities or departments to schedule patients for tests or admission.',
        '2039.0': 'Inventory and order medical, lab, or office supplies or equipment.',
        '20391.0': 'Identify risks for natural disasters, such as mudslides, earthquakes, or volcanic eruptions.',
        '20395.0': 'Monitor and control temperature of products.',
        '20461.0': 'Review and analyze legislation, laws, or public policy and recommend changes to promote or support interests of the general population or special groups.',
        '20483.0': 'Develop final construction plans that include aesthetic representations of the structure or details for its construction.',
        '20486.0': 'Plan layouts of structural architectural projects.',
        '20488.0': 'Create three-dimensional or interactive representations of designs, using computer-assisted design software.',
        '20497.0': 'Operate computer-assisted engineering or design software or equipment to perform electronics engineering tasks.',
        '20526.0': 'Measure or weigh ingredients used in laboratory testing.',
        '20558.0': 'Transfer photographs to computers for editing, archiving, and electronic transmission.',
        '20578.0': 'Listen and provide emotional support and encouragement to psychiatric patients.',
        '20608.0': 'Prepare and submit sales contracts for orders.',
        '20609.0': 'Contact new or existing customers to discuss how specific products or services can meet their needs.',
        '20610.0': 'Select or assist customers in selecting products based on customer needs, product specifications, and applicable regulations.',
        '20612.0': 'Identify prospective customers, using business directories, leads from existing clients, participation in organizations, or trade show or conference attendance.',
        '20614.0': 'Study documentation or other information for new scientific or technical products.',
        '20615.0': 'Attend sales or trade meetings or read related publications to obtain information about market conditions, business trends, environmental regulations, or industry developments.',
        '20618.0': 'Initiate sales campaigns to meet sales and production expectations.',
        '20620.0': 'Verify customer credit ratings.',
        '20695.0': 'Operate equipment, such as truck cab computers, CB radios, phones, or global positioning systems (GPS) equipment to exchange necessary information with bases, supervisors, or other drivers.',
        '20704.0': 'Establish or implement departmental policies, goals, objectives, or procedures in conjunction with board members, organization officials, or staff members.',
        '20728.0': 'Direct aerospace research and development programs.',
        '20733.0': 'Research, design, evaluate, install, operate, or maintain mechanical products, equipment, systems or processes to meet requirements.',
        '20744.0': 'Conserve and preserve manuscripts, records, and other artifacts.',
        '20761.0': 'Operate vehicles or powered equipment, such as mowers, tractors, twin-axle vehicles, snow blowers, chainsaws, electric clippers, sod cutters, or pruning saws.',
        '20776.0': 'Assist customers, such as responding to customer complaints and updating them about back-ordered parts.',
        '20826.0': 'Start agitators, shakers, conveyors, pumps, or centrifuge machines.',
        '20842.0': 'Monitor employees\' work schedules and attendance for payroll purposes.',
        '20843.0': 'Inspect government property, such as construction sites or public housing, to ensure compliance with contract specifications or legal requirements.',
        '20844.0': 'Investigate alleged license or permit violations.',
        '20845.0': 'Investigate applications for special licenses or permits.',
        '20855.0': 'Conduct energy audits to evaluate energy use and to identify conservation and cost reduction measures.',
        '20877.0': 'Prepare, draft, and review legal documents, such as wills, deeds, patent applications, mortgages, leases, and contracts.',
        '209.0': 'Conduct public educational programs on forest care and conservation.',
        '20900.0': 'Record and maintain information on clients, vendors, and travel packages.',
        '20950.0': 'Troubleshoot program and system malfunctions to restore normal functioning.',
        '20960.0': 'Identify opportunities or implement changes to improve manufacturing processes or products or to reduce costs, using knowledge of fabrication processes, tooling and production equipment, assembly methods, quality control standards, or product design, materials and parts.',
        '20964.0': 'Produce three-dimensional models, using computer-aided design (CAD) software.',
        '21000.0': 'Write text, such as stories, articles, editorials, or newsletters.',
        '21049.0': 'Develop Web sites.',
        '21062.0': 'Investigate facts and law of cases and search pertinent sources, such as public records and internet sources, to determine causes of action and to prepare cases.',
        '21065.0': 'Enter information about museum collections into computer databases.',
        '21097.0': 'Administer employee benefit plans.',
        '21144.0': 'Educate the public about fire safety and prevention.',
        '21193.0': 'Instruct patients in proper body mechanics and in ways to improve functional mobility, such as aquatic exercise.',
        '2125.0': 'Package, store and retrieve evidence.',
        '21271.0': 'Supervise administrative staff and provide training and orientation to new staff.',
        '21276.0': 'Monitor the facility to ensure that it remains safe, secure, and well-maintained.',
        '21283.0': 'Analyze and evaluate security operations to identify risks or opportunities for improvement through auditing, review, or assessment.',
        '21284.0': 'Assess risks to mitigate potential consequences of incidents and develop a plan to respond to incidents.',
        '21286.0': 'Communicate security status, updates, and actual or potential problems, using established protocols.',
        '21288.0': 'Conduct threat or vulnerability analyses to determine probable frequency, criticality, consequence, or severity of natural or man-made disasters or criminal activity on the organization\'s profitability or delivery of products or services.',
        '21292.0': 'Develop or manage investigation programs, including collection and preservation of video and notes of surveillance processes or investigative interviews.',
        '21298.0': 'Identify, investigate, or resolve security breaches.',
        '213.0': 'Monitor wildlife populations and assess the impacts of forest operations on population and habitats.',
        '21302.0': 'Plan, direct, or coordinate security activities to safeguard company employees, guests, or others on company property.',
        '21305.0': 'Respond to medical emergencies, bomb threats, fire alarms, or intrusion alarms, following emergency response procedures.',
        '21311.0': 'Establish and maintain relationships with individual or business customers or provide assistance with problems these customers may encounter.',
        '21317.0': 'Examine, evaluate, or process loan applications.',
        '21318.0': 'Approve, reject, or coordinate the approval or rejection of lines of credit or commercial, real estate, or personal loans.',
        '21319.0': 'Oversee the flow of cash or financial instruments.',
        '21320.0': 'Prepare financial or regulatory reports required by laws, regulations, or boards of directors.',
        '21323.0': 'Evaluate financial reporting systems, accounting or collection procedures, or investment activities and make recommendations for changes to procedures, operating systems, budgets, or other financial control functions.',
        '21326.0': 'Review collection reports to determine the status of collections and the amounts of outstanding balances.',
        '21344.0': 'Collaborate with other departments to integrate logistics with business systems or processes, such as customer sales, order management, accounting, or shipping.',
        '21346.0': 'Resolve problems concerning transportation, logistics systems, imports or exports, or customer issues.',
        '21357.0': 'Analyze expenditures and other financial information to develop plans, policies, or budgets for increasing profits or improving services.',
        '21370.0': 'Collect and record growth, production, and environmental data.',
        '2139.0': 'Operate detecting devices to screen individuals and prevent passage of prohibited articles into restricted areas.',
        '214.0': 'Plan and direct construction and maintenance of recreation facilities, fire towers, trails, roads and bridges, ensuring that they comply with guidelines and regulations set for forested public lands.',
        '21408.0': 'Plan programs of events or schedules of activities.',
        '21438.0': 'Examine titles to property to determine validity and act as company agent in transactions with property owners.',
        '21442.0': 'Obtain credit information from banks and other credit services.',
        '21447.0': 'Evaluate applications, records, or documents to gather information about eligibility or liability issues.',
        '21461.0': 'Verify that all firm and regulatory policies and procedures have been documented, implemented, and communicated.',
        '21463.0': 'Communicate with key stakeholders to determine project requirements and objectives.',
        '21466.0': 'Develop or update project plans including information such as objectives, technologies, schedules, funding, and staffing.',
        '21467.0': 'Identify project needs such as resources, staff, or finances by reviewing project objectives and schedules.',
        '21470.0': 'Monitor project milestones and deliverables.',
        '21471.0': 'Monitor the performance of project team members to provide performance feedback.',
        '21473.0': 'Plan, schedule, or coordinate project activities to meet deadlines.',
        '21475.0': 'Produce and distribute project documents.',
        '21476.0': 'Propose, review, or approve modifications to project plans.',
        '21479.0': 'Request and review project updates to ensure deadlines are met.',
        '21480.0': 'Schedule or facilitate project meetings.',
        '21505.0': 'Prepare detailed reports on audit findings.',
        '21507.0': 'Collect and analyze data to detect deficient controls, duplicated effort, extravagance, fraud, or non-compliance with laws, regulations, and management policies.',
        '21510.0': 'Confer with company officials about financial and regulatory matters.',
        '21511.0': 'Examine and evaluate financial and information systems, recommending controls to ensure system reliability and data integrity.',
        '21512.0': 'Inspect cash on hand, notes receivable and payable, negotiable securities, and canceled checks to confirm records are accurate.',
        '21513.0': 'Examine records and interview workers to ensure recording of transactions and compliance with laws and regulations.',
        '21514.0': 'Prepare, examine, or analyze accounting records, financial statements, or other financial reports to assess accuracy, completeness, and conformance to reporting and procedural standards.',
        '21515.0': 'Prepare adjusting journal entries.',
        '21516.0': 'Review accounts for discrepancies and reconcile differences.',
        '21517.0': 'Establish tables of accounts and assign entries to proper accounts.',
        '21518.0': 'Examine inventory to verify journal and ledger entries.',
        '21519.0': 'Analyze business operations, trends, costs, revenues, financial commitments, and obligations to project future revenues and expenses or to provide advice.',
        '21520.0': 'Report to management regarding the finances of establishment.',
        '21521.0': 'Develop, implement, modify, and document recordkeeping and accounting systems, making use of current computer technology.',
        '21523.0': 'Examine whether the organization\'s objectives are reflected in its management activities, and whether employees understand the objectives.',
        '21529.0': 'Direct activities of personnel engaged in filing, recording, compiling, and transmitting financial records.',
        '21532.0': 'Prepare, analyze, or verify annual reports, financial statements, and other records, using accepted accounting and statistical procedures to assess financial condition and facilitate financial planning.',
        '21533.0': 'Process invoices for payment.',
        '21534.0': 'Review data about material assets, net worth, liabilities, capital stock, surplus, income, or expenditures.',
        '21651.0': 'Test changes to database applications or systems.',
        '21652.0': 'Develop data model describing data elements and their use, following procedures and using pen, template or computer software.',
        '21653.0': 'Develop methods for integrating different products so they work properly together, such as customizing commercial databases to fit specific needs.',
        '21654.0': 'Establish and calculate optimum values for database parameters, using manuals and calculators.',
        '21656.0': 'Review project requests describing database user needs to estimate time and cost required to accomplish project.',
        '21657.0': 'Test programs or databases, correct errors, and make necessary modifications.',
        '21661.0': 'Analyze information to determine, recommend, and plan installation of a new system or modification of an existing system.',
        '21662.0': 'Analyze user needs and software requirements to determine feasibility of design within time and cost constraints.',
        '21664.0': 'Confer with systems analysts, engineers, programmers and others to design systems and to obtain information on project limitations and capabilities, performance requirements and interfaces.',
        '21665.0': 'Consult with customers or other departments on project status, proposals, or technical issues, such as software system design or maintenance.',
        '21666.0': 'Coordinate installation of software system.',
        '21667.0': 'Design, develop and modify software systems, using scientific analysis and mathematical models to predict and measure outcomes and consequences of design.',
        '21669.0': 'Develop or direct software system testing or validation procedures, programming, or documentation.',
        '21670.0': 'Modify existing software to correct errors, adapt it to new hardware, or upgrade interfaces and improve performance.',
        '21672.0': 'Obtain and evaluate information on factors such as reporting formats required, costs, or security needs to determine hardware configuration.',
        '21675.0': 'Specify power supply requirements and configuration.',
        '21676.0': 'Store, retrieve, and manipulate data for analysis of system capabilities and requirements.',
        '21690.0': 'Create searchable indices for Web page content.',
        '21691.0': 'Create Web models or prototypes that include physical, interface, logical, or data models.',
        '21693.0': 'Develop and document style guidelines for Web site content.',
        '21694.0': 'Develop new visual design concepts and modify concepts based on stakeholder feedback.',
        '21697.0': 'Develop Web site maps, application models, image templates, or page templates that meet project goals, user needs, or industry standards.',
        '21699.0': 'Direct and execute pre-production activities, such as creating moodboards or storyboards and establishing a project timeline.',
        '2170.0': 'Season and cook food according to recipes or personal judgment and experience.',
        '21705.0': 'Perform or direct Web site updates.',
        '21706.0': 'Perform Web site tests according to planned schedules, or after any Web site or product revision.',
        '21714.0': 'Write supporting code for Web applications or Web sites.',
        '21715.0': 'Produce data layers, maps, tables, or reports, using spatial analysis procedures or Geographic Information Systems (GIS) technology, equipment, or systems.',
        '21716.0': 'Design or prepare graphic representations of Geographic Information Systems (GIS) data, using GIS hardware or software applications.',
        '21717.0': 'Maintain or modify existing Geographic Information Systems (GIS) databases.',
        '21718.0': 'Provide technical expertise in Geographic Information Systems (GIS) technology to clients or users.',
        '21719.0': 'Perform computer programming, data analysis, or software development for Geographic Information Systems (GIS) applications, including the maintenance of existing systems or research and development for future enhancements.',
        '21720.0': 'Enter data into Geographic Information Systems (GIS) databases, using techniques such as coordinate geometry, keyboard entry of tabular data, manual digitizing of maps, scanning or automatic conversion to vectors, or conversion of other sources of digital data.',
        '21721.0': 'Review existing or incoming data for currency, accuracy, usefulness, quality, or completeness of documentation.',
        '21722.0': 'Perform geospatial data building, modeling, or analysis, using advanced spatial analysis, data manipulation, or cartography software.',
        '21727.0': 'Collect, compile, or integrate Geographic Information Systems (GIS) data, such as remote sensing or cartographic data for inclusion in map manuscripts.',
        '21734.0': 'Develop specialized computer software routines, internet-based Geographic Information Systems (GIS) databases, or business applications to customize geographic information.',
        '21745.0': 'Collect stakeholder data to evaluate risk and to develop mitigation strategies.',
        '21746.0': 'Conduct network and security system audits, using established criteria.',
        '21747.0': 'Configure information systems to incorporate principles of least functionality and least access.',
        '21749.0': 'Develop and execute tests that simulate the techniques of known cyber threat actors.',
        '21750.0': 'Develop infiltration tests that exploit device vulnerabilities.',
        '21752.0': 'Develop security penetration testing processes, such as wireless, data networks, and telecommunication security tests.',
        '21754.0': 'Document penetration test findings.',
        '21758.0': 'Identify security system weaknesses, using penetration tests.',
        '21763.0': 'Test the security of systems by attempting to gain access to networks, Web-based applications, or computers.',
        '21769.0': 'Coordinate monitoring of networks or systems for security breaches or intrusions.',
        '21770.0': 'Coordinate vulnerability assessments or analysis of information security systems.',
        '21771.0': 'Develop information security standards and best practices.',
        '21772.0': 'Develop or implement software tools to assist in the detection, prevention, and analysis of security threats.',
        '21773.0': 'Develop or install software, such as firewalls and data encryption programs, to protect sensitive information.',
        '21778.0': 'Oversee performance of risk assessment or execution of system tests to ensure the functioning of data processing activities or security measures.',
        '21781.0': 'Review security assessments for computing environments or check for compliance with cybersecurity standards and regulations.',
        '21782.0': 'Scan networks, using vulnerability assessment tools to identify vulnerabilities.',
        '21787.0': 'Analyze log files or other digital information to identify the perpetrators of network intrusions.',
        '21788.0': 'Conduct predictive or reactive analyses on security measures to support cyber security initiatives.',
        '21789.0': 'Create system images or capture network settings from information technology environments to preserve as evidence.',
        '21790.0': 'Develop plans for investigating alleged computer crimes, violations, or suspicious activity.',
        '21791.0': 'Develop policies or requirements for data collection, processing, or reporting.',
        '21792.0': 'Duplicate digital evidence to use for data recovery and analysis procedures.',
        '21793.0': 'Identify or develop reverse-engineering tools to improve system capabilities or detect vulnerabilities.',
        '21794.0': 'Maintain cyber defense software or hardware to support responses to cyber incidents.',
        '21796.0': 'Perform file signature analysis to verify files on storage media or discover potential hidden files.',
        '21797.0': 'Perform forensic investigations of operating or file systems.',
        '21798.0': 'Perform web service network traffic analysis or waveform analysis to detect anomalies, such as unusual events or trends.',
        '21799.0': 'Preserve and maintain digital forensic evidence for analysis.',
        '21800.0': 'Recommend cyber defense software or hardware to support responses to cyber incidents.',
        '21801.0': 'Recover data or decrypt seized data.',
        '21802.0': 'Write and execute scripts to automate tasks, such as parsing large data files.',
        '21805.0': 'Write technical summaries to report findings.',
        '21806.0': 'Assess blockchain threats, such as untested code and unprotected keys.',
        '21819.0': 'Implement catastrophic failure handlers to identify security breaches and prevent serious damage.',
        '21823.0': 'Analyze, manipulate, or process large sets of data using statistical software.',
        '21824.0': 'Apply feature selection algorithms to models predicting outcomes of interest, such as sales, attrition, and healthcare use.',
        '21826.0': 'Clean and manipulate raw data using statistical software.',
        '21827.0': 'Compare models using statistical performance metrics, such as loss functions or proportion of explained variance.',
        '21828.0': 'Create graphs, charts, or other visualizations to convey the results of data analysis using specialized software.',
        '21832.0': 'Identify relationships and trends or any factors that could affect the results of research.',
        '21834.0': 'Propose solutions in engineering, the sciences, and other fields using mathematical theories and techniques.',
        '21837.0': 'Test, validate, and reformulate models to ensure accurate prediction of outcomes of interest.',
        '21838.0': 'Write new functions or applications in programming languages to conduct analyses.',
        '21844.0': 'Consult with chemists or biologists to develop or evaluate novel technologies.',
        '21851.0': 'Develop statistical models or simulations, using statistical or modeling software.',
        '21857.0': 'Read current scientific or trade literature to stay abreast of scientific, industrial, or technological advances.',
        '21917.0': 'Produce drawings, using computer-assisted drafting systems (CAD) or drafting machines, or by hand, using compasses, dividers, protractors, triangles, and other drafting devices.',
        '21918.0': 'Draft plans and detailed drawings for structures, installations, and construction projects, such as highways, sewage disposal systems, and dikes, working from sketches or notes.',
        '21920.0': 'Analyze building codes, by-laws, space and site requirements, and other technical documents and reports to determine their effect on architectural designs.',
        '21932.0': 'Obtain and assemble data to complete architectural designs, visiting job sites to compile measurements as necessary.',
        '21938.0': 'Calculate weights, volumes, and stress factors and their implications for technical aspects of designs.',
        '21978.0': 'Read blueprints, wiring diagrams, schematic drawings, or engineering instructions for assembling electronics units, applying knowledge of electronic theory and components.',
        '21998.0': 'Install or maintain electrical control systems, industrial automation systems, or electrical equipment, including control circuits, variable speed drives, or programmable logic controllers.',
        '22014.0': 'Install or program computer hardware or machine or instrumentation software in microprocessor-based systems.',
        '22165.0': 'Document patient information including session notes, progress notes, recommendations, and treatment plans.',
        '22193.0': 'Conduct neuropsychological evaluations such as assessments of intelligence, academic ability, attention, concentration, sensorimotor function, language, learning, and memory.',
        '22194.0': 'Conduct research on neuropsychological disorders.',
        '22232.0': 'Collect information and make judgments through observation, interviews, and review of documents.',
        '22246.0': 'Create data records for use in describing and analyzing social patterns and processes, using photography, videography, and audio recordings.',
        '22251.0': 'Clean, restore, and preserve artifacts.',
        '22253.0': 'Organize public exhibits and displays to promote public awareness of diverse and distinctive cultural traditions.',
        '22263.0': 'Collect or prepare solid or fluid samples for analysis.',
        '22285.0': 'Collaborate with hydrogeologists to evaluate groundwater or well circulation.',
        '22291.0': 'Evaluate and interpret seismic data with the aid of computers.',
        '22397.0': 'Grade students\' assignments and exams.',
        '22403.0': 'Take class attendance and maintain attendance records.',
        '22411.0': 'Search standard reference materials, including online sources and the Internet, to answer patrons\' reference questions.',
        '22426.0': 'Engage in professional development activities, such as taking continuing education classes and attending or participating in conferences, workshops, professional meetings, and associations.',
        '22432.0': 'Set up, adjust, and operate audio-visual equipment, such as cameras, film and slide projectors, and recording equipment, for meetings, events, classes, seminars, and video conferences.',
        '22438.0': 'Analyze performance data to determine effectiveness of instructional systems, courses, or instructional materials.',
        '22440.0': 'Conduct needs assessments and strategic learning assessments to develop the basis for curriculum development or to update curricula.',
        '22544.0': 'Use gestures to shape the music being played, communicating desired tempo, phrasing, tone, color, pitch, volume, and other performance aspects.',
        '22545.0': 'Direct groups at rehearsals and live or recorded performances to achieve desired effects such as tonal and harmonic balance dynamics, rhythm, and tempo.',
        '22546.0': 'Study scores to learn the music in detail, and to develop interpretations.',
        '22547.0': 'Apply elements of music theory to create musical and tonal structures, including harmonies and melodies.',
        '22548.0': 'Consider such factors as ensemble size and abilities, availability of scores, and the need for musical variety, to select music to be performed.',
        '22549.0': 'Determine voices, instruments, harmonic structures, rhythms, tempos, and tone balances required to achieve the effects desired in a musical composition.',
        '22550.0': 'Experiment with different sounds, and types and pieces of music, using synthesizers and computers as necessary to test and evaluate ideas.',
        '22551.0': 'Transcribe ideas for musical compositions into musical notation, using instruments, pen and paper, or computers.',
        '22552.0': 'Audition and select performers for musical presentations.',
        '22553.0': 'Plan and schedule rehearsals and performances, and arrange details such as locations, accompanists, and instrumentalists.',
        '22554.0': 'Write musical scores for orchestras, bands, choral groups, or individual instrumentalists or vocalists, using knowledge of music theory and of instrumental and vocal capabilities.',
        '22555.0': 'Position members within groups to obtain balance among instrumental or vocal sections.',
        '22561.0': 'Write music for commercial mediums, including advertising jingles or film soundtracks.',
        '22562.0': 'Transpose music from one voice or instrument to another to accommodate particular musicians.',
        '22563.0': 'Rewrite original musical scores in different musical styles by changing rhythms, harmonies, or tempos.',
        '22564.0': 'Arrange music composed by others, changing the music to achieve desired effects.',
        '22566.0': 'Study films or scripts to determine how musical scores can be used to create desired effects or moods.',
        '22570.0': 'Copy parts from scores for individual performers.',
        '22572.0': 'Produce recordings of music.',
        '22573.0': 'Stay abreast of the latest trends in music and music technology.',
        '22580.0': 'Memorize musical selections and routines, or sing following printed text, musical notation, or customer instructions.',
        '22581.0': 'Play musical instruments as soloists, or as members or guest artists of musical groups such as orchestras, ensembles, or bands.',
        '22585.0': 'Listen to recordings to master pieces or to maintain and improve skills.',
        '22588.0': 'Audition for orchestras, bands, or other musical groups.',
        '22589.0': 'Seek out and learn new music suitable for live performance or recording.',
        '22590.0': 'Make or participate in recordings in music studios.',
        '22592.0': 'Make or participate in recordings.',
        '22596.0': 'Direct bands or orchestras.',
        '22598.0': 'Arrange and edit music to fit style and purpose.',
        '22630.0': 'Gather information and develop perspectives about news subjects through research, interviews, observation, and experience.',
        '22642.0': 'Check reference materials, such as books, news files, or public records, to obtain relevant facts.',
        '22657.0': 'Write articles, bulletins, sales letters, speeches, and other related informative, marketing and promotional material.',
        '22659.0': 'Invent names for products and write the slogans that appear on packaging, brochures and other promotional material.',
        '22670.0': 'Write advertising material for use by publication, broadcast, or internet media to promote the sale of goods and services.',
        '22693.0': 'Compare measurements of heart wall thickness and chamber sizes to standards to identify abnormalities, using the results of an echocardiogram.',
        '22694.0': 'Conduct electrocardiogram (EKG), phonocardiogram, echocardiogram, or other cardiovascular tests to record patients\' cardiac activity, using specialized electronic test equipment, recording devices, or laboratory instruments.',
        '22696.0': 'Conduct research to develop or test medications, treatments, or procedures that prevent or control disease or injury.',
        '22700.0': 'Diagnose medical conditions of patients, using records, reports, test results, or examination information.',
        '22704.0': 'Monitor patients\' conditions and progress, and reevaluate treatments, as necessary.',
        '22717.0': 'Collect and record patient information, such as medical history or examination results, in electronic or handwritten medical records.',
        '22728.0': 'Refer patients to specialists or other practitioners.',
        '22729.0': 'Select and prescribe medications to address patient needs.',
        '22751.0': 'Examine patient to obtain information on medical condition and surgical risk.',
        '2277.0': 'Write patrons\' food orders on order slips, memorize orders, or enter orders into computers for transmittal to kitchen staff.',
        '22782.0': 'Examine slides under microscopes to ensure tissue preparation meets laboratory requirements.',
        '22791.0': 'Stain tissue specimens with dyes or other chemicals to make cell details visible under microscopes.',
        '2283.0': 'Present menus to patrons and answer questions about menu items, making recommendations upon request.',
        '22878.0': 'Compile and maintain patients\' medical records to document condition and treatment and to provide data for research or cost control and care improvement efforts.',
        '22880.0': 'Enter data, such as demographic characteristics, history and extent of disease, diagnostic procedures, or treatment into computer.',
        '22881.0': 'Identify, compile, abstract, and code patient data, using standard classification systems.',
        '22886.0': 'Protect the security of medical records to ensure that confidentiality is maintained.',
        '22888.0': 'Resolve or clarify codes or diagnoses with conflicting, missing, or unclear information by consulting with doctors or others or by participating in the coding team\'s regular meetings.',
        '22889.0': 'Retrieve patient medical records for physicians, technicians, or other medical personnel.',
        '2289.0': 'Perform food preparation duties, such as preparing salads, appetizers, and cold dishes, portioning desserts, and brewing coffee.',
        '22890.0': 'Review records for completeness, accuracy, and compliance with regulations.',
        '22896.0': 'Design databases to support healthcare applications, ensuring security, performance and reliability.',
        '22898.0': 'Evaluate and recommend upgrades or improvements to existing computerized healthcare systems.',
        '22929.0': 'Perform administrative duties, such as compiling and maintaining records, completing forms, preparing reports, or composing correspondence.',
        '2301.0': 'Receive and record patrons\' dining reservations.',
        '23024.0': 'Record progress of investigation, maintain informational files on suspects, and submit reports to commanding officer or magistrate to authorize warrants.',
        '23030.0': 'Examine records and governmental agency files to find identifying data about suspects.',
        '23034.0': 'Obtain and verify evidence by interviewing and observing suspects and witnesses or by analyzing records.',
        '23045.0': 'Search for and collect evidence, such as fingerprints, using investigative equipment.',
        '23048.0': 'Collaborate with other offices and agencies to exchange information and coordinate activities.',
        '23052.0': 'Provide for public safety by maintaining order, responding to emergencies, protecting people and property, enforcing motor vehicle and criminal laws, and promoting good community relations.',
        '23053.0': 'Record facts to prepare reports that document incidents and activities.',
        '23057.0': 'Monitor, note, report, and investigate suspicious persons and situations, safety hazards, and unusual or illegal activity in patrol area.',
        '23061.0': 'Relay complaint and emergency-request information to appropriate agency dispatchers.',
        '23074.0': 'Inform citizens of community services and recommend options to facilitate longer-term problem resolution.',
        '23078.0': 'Supervise law enforcement staff, such as jail staff, officers, and deputy sheriffs.',
        '23099.0': 'Communicate with customers regarding orders, comments, and complaints.',
        '2315.0': 'Supply guests or travelers with directions, travel information, and other information, such as available services and points of interest.',
        '23192.0': 'Make bids or offers to buy or sell securities.',
        '23193.0': 'Monitor markets or positions.',
        '23194.0': 'Agree on buying or selling prices at optimal levels for clients.',
        '23195.0': 'Keep accurate records of transactions.',
        '23196.0': 'Buy or sell stocks, bonds, commodity futures, foreign currencies, or other securities on behalf of investment dealers.',
        '23197.0': 'Complete sales order tickets and submit for processing of client-requested transactions.',
        '23198.0': 'Report all positions or trading results.',
        '23199.0': 'Interview clients to determine clients\' assets, liabilities, cash flow, insurance coverage, tax status, or financial objectives.',
        '23203.0': 'Identify opportunities or develop channels for purchase or sale of securities or commodities.',
        '23204.0': 'Develop financial plans, based on analysis of clients\' financial status.',
        '23205.0': 'Review all securities transactions to ensure accuracy of information and conformance to governing agency regulations.',
        '23206.0': 'Contact prospective customers to present information and explain available services.',
        '23207.0': 'Devise trading, option, or hedge strategies.',
        '23208.0': 'Track and analyze factors that affect price movement, such as trade policies, weather conditions, political developments, or supply and demand changes.',
        '23209.0': 'Inform other traders, managers, or customers of market conditions, including volume, price, competition, or dynamics.',
        '23210.0': 'Offer advice on the purchase or sale of particular securities.',
        '23213.0': 'Calculate costs for billings or commissions.',
        '23214.0': 'Prepare financial reports to monitor client or corporate finances.',
        '23215.0': 'Supply the latest price quotes on any security, as well as information on the activities or financial positions of the corporations issuing these securities.',
        '23219.0': 'Prepare and send requests for price quotations to all companies in a particular market.',
        '23220.0': 'Price securities or commodities based on market conditions.',
        '23221.0': 'Purchase or sell financial derivatives for customers.',
        '23231.0': 'Identify prospective customers using business directories, leads from clients, or information from conferences or trade shows.',
        '23233.0': 'Maintain customer records using automated systems.',
        '23234.0': 'Monitor market conditions, innovations, and competitors\' services, prices, and sales.',
        '23239.0': 'Verify signatures and required information on checks.',
        '23270.0': 'Question applicants to obtain required information, such as name, address, or age, and record data on prescribed forms.',
        '23271.0': 'Issue public notification of all official activities or meetings.',
        '23274.0': 'Prepare meeting agendas or packets of related information.',
        '23285.0': 'Respond to requests for information from the public, other municipalities, state officials, or state and federal legislative offices.',
        '23293.0': 'Participate in the administration of municipal elections, such as preparation or distribution of ballots, appointment or training of election officers, or tabulation or certification of results.',
        '23294.0': 'Issue various permits and licenses, such as marriage, fishing, hunting, and dog licenses, and collect appropriate fees.',
        '23298.0': 'Compile and analyze credit information gathered by investigation.',
        '23299.0': 'Obtain information about potential creditors from banks, credit bureaus, and other credit services, and provide reciprocal information if requested.',
        '23301.0': 'Evaluate customers\' computerized credit records and payment histories to decide whether to approve new credit, based on predetermined standards.',
        '23309.0': 'Consult with customers to resolve complaints or verify financial or credit transactions.',
        '2354.0': 'Help prepare and serve nutritionally balanced meals and snacks for children.',
        '23556.0': 'Install, connect, or adjust thermostats, humidistats, or timers.',
        '23558.0': 'Study blueprints, design specifications, or manufacturers\' recommendations to ascertain the configuration of heating or cooling equipment components and to ensure the proper installation of components.',
        '23562.0': 'Inspect and test systems to verify system compliance with plans and specifications or to detect and locate malfunctions.',
        '23565.0': 'Adjust system controls to settings recommended by manufacturer to balance system.',
        '23579.0': 'Install or repair self-contained ground source heat pumps or hybrid ground or air source heat pumps to minimize carbon-based energy consumption and reduce carbon emissions.',
        '23580.0': 'Repair or service heating, ventilating, and air conditioning (HVAC) systems to improve efficiency, such as by changing filters, cleaning ducts, and refilling non-toxic refrigerants.',
        '23624.0': 'Clean and polish metal items and jewelry pieces, using jewelers\' tools, polishing wheels, and chemical baths.',
        '2372.0': 'Perform housekeeping duties, such as cooking, cleaning, washing clothes or dishes, or running errands.',
        '2373.0': 'Care for individuals or families during periods of incapacitation, family disruption, or convalescence, providing companionship, personal care, or help in adjusting to new lifestyles.',
        '23733.0': 'Check the condition of a vehicle\'s tires, brakes, windshield wipers, lights, oil, fuel, water, and safety equipment to ensure that everything is in working order.',
        '23735.0': 'Communicate with dispatchers by radio, telephone, or computer to exchange information and receive requests for passenger service.',
        '23741.0': 'Notify dispatchers or company mechanics of vehicle problems.',
        '23745.0': 'Perform routine vehicle maintenance, such as regulating tire pressure and adding gasoline, oil, and water.',
        '23752.0': 'Record vehicle routes.',
        '23754.0': 'Report any vehicle malfunctions or needed repairs.',
        '23759.0': 'Communicate with dispatchers by radio, telephone, or computer to exchange information and receive requests for passenger service.',
        '23762.0': 'Drive taxicabs or privately owned vehicles to transport passengers.',
        '2382.0': 'Manage the daily operations of recreational facilities.',
        '23878.0': 'Receive and count stock items, and record data manually or on computer.',
        '23885.0': 'Take inventory or examine merchandise to identify items to be reordered or replenished.',
        '23886.0': 'Issue or distribute materials, products, parts, and supplies to customers or coworkers, based on information from incoming requisitions.',
        '23898.0': 'Determine proper storage methods, identification, and stock location, based on turnover, environmental factors, and physical capabilities of facilities.',
        '23905.0': 'Prepare or review reports, manuscripts, or meeting presentations.',
        '2392.0': 'Schedule maintenance and use of facilities.',
        '23947.0': 'Create mechanical models to simulate mechatronic design concepts.',
        '23958.0': 'Conduct literature reviews.',
        '2396.0': 'Encourage participants to develop their own activities and leadership skills through group discussions.',
        '23967.0': 'Examine physical evidence, such as hair, biological fluids, fiber, wood, or soil residues to obtain information about its source and composition.',
        '23969.0': 'Analyze data from computers or other digital media sources for evidence related to criminal activity.',
        '23970.0': 'Prepare digital files for printing.',
        '23985.0': 'Develop and administer compensation programs, such as merit or incentive pay.',
        '23994.0': 'Prepare, edit, or review legal documents, including legislation, briefs, pleadings, appeals, wills, contracts, and real estate closing statements.',
        '2400.0': 'Receive payment by cash, check, credit cards, vouchers, or automatic debits.',
        '2401.0': 'Issue receipts, refunds, credits, or change due to customers.',
        '2405.0': 'Establish or identify prices of goods, services, or admission, and tabulate bills, using calculators, cash registers, or optical price scanners.',
        '2408.0': 'Answer customers\' questions, and provide information on procedures or policies.',
        '2411.0': 'Calculate total payments received during a time period, and reconcile this with total sales.',
        '2412.0': 'Compute and record totals of transactions.',
        '2487.0': 'Operate computers programmed with accounting software to record, store, and analyze information.',
        '2489.0': 'Debit, credit, and total accounts on computer spreadsheets and databases, using specialized accounting software.',
        '2490.0': 'Classify, record, and summarize numerical and financial data to compile and keep financial records, using journals and ledgers or computers.',
        '2491.0': 'Calculate, prepare, and issue bills, invoices, account statements, and other financial statements according to established procedures.',
        '2492.0': 'Compile statistical, financial, accounting, or auditing reports and tables pertaining to such matters as cash receipts, expenditures, accounts payable and receivable, and profits and losses.',
        '2494.0': 'Access computerized financial information to answer general questions as well as those related to specific accounts.',
        '2496.0': 'Reconcile or note and report discrepancies found in records.',
        '2497.0': 'Perform financial calculations, such as amounts due, interest charges, balances, discounts, equity, and principal.',
        '2500.0': 'Receive, record, and bank cash, checks, and vouchers.',
        '2501.0': 'Calculate and prepare checks for utilities, taxes, and other payments.',
        '2503.0': 'Reconcile records of bank transactions.',
        '2504.0': 'Prepare trial balances of books.',
        '2511.0': 'Maintain inventory records.',
        '2550.0': 'Cash checks and pay out money after verifying that signatures are correct, that written and numerical amounts agree, and that accounts have sufficient funds.',
        '2553.0': 'Enter customers\' transactions into computers to record transactions and issue computer-generated receipts.',
        '2558.0': 'Process transactions, such as term deposits, retirement savings plan contributions, automated teller transactions, night deposits, and mail deposits.',
        '2560.0': 'Resolve problems or discrepancies concerning customers\' accounts.',
        '2563.0': 'Monitor bank vaults to ensure cash balances are correct.',
        '2567.0': 'Process and maintain records of customer loans.',
        '2571.0': 'Obtain and process information required for the provision of services, such as opening accounts, savings plans, and purchasing bonds.',
        '2578.0': 'Keep records of customer interactions or transactions, recording details of inquiries, complaints, or comments, as well as actions taken.',
        '2579.0': 'Resolve customers\' service or billing complaints by performing activities such as exchanging merchandise, refunding money, or adjusting bills.',
        '2581.0': 'Contact customers to respond to inquiries or to notify them of claim investigation results or any planned adjustments.',
        '2582.0': 'Refer unresolved customer grievances to designated departments for further investigation.',
        '2584.0': 'Complete contract forms, prepare change of address records, or issue service discontinuance orders, using computers.',
        '2612.0': 'Keep records of room availability and guests\' accounts, manually or using computers.',
        '2618.0': 'Transmit and receive messages, using telephones or telephone switchboards.',
        '2620.0': 'Make and confirm reservations.',
        '2624.0': 'Arrange tours, taxis, or restaurant reservations for customers.',
        '2636.0': 'Perform patient services, such as answering the telephone or assisting patients with financial or medical questions.',
        '2671.0': 'Receive and respond to customer complaints.',
        '2680.0': 'Recommend merchandise or services that will meet customers\' needs.',
        '2681.0': 'Adjust inventory records to reflect product movement.',
        '2689.0': 'Record data for each employee, including such information as addresses, weekly earnings, absences, amount of sales or production, supervisory reports on performance, and dates of and reasons for terminations.',
        '2692.0': 'Examine employee files to answer inquiries and provide information for personnel actions.',
        '2700.0': 'Arrange for in-house and external training activities.',
        '2723.0': 'Schedule or dispatch workers, work crews, equipment, or service vehicles to appropriate locations, according to customer requests, specifications, or needs, using radios or telephones.',
        '2790.0': 'Answer telephones and give information to callers, take messages, or transfer calls to appropriate individuals.',
        '2817.0': 'View monitors for visual representation of work in progress and for instructions and feedback throughout process, making modifications as necessary.',
        '2820.0': 'Position text and art elements from a variety of databases in a visually appealing way to design print or web pages, using knowledge of type styles and size and layout patterns.',
        '2823.0': 'Import text and art elements, such as electronic clip art or electronic files from photographs that have been scanned or produced with a digital camera, using computer software.',
        '2824.0': 'Prepare sample layouts for approval, using computer software.',
        '2831.0': 'Collaborate with graphic artists, editors and writers to produce master copies according to design specifications.',
        '2832.0': 'Create special effects such as vignettes, mosaics, and image combining, and add elements such as sound and animation to electronic publications.',
        '286.0': 'Manage own accounts and projects, working within budget and scheduling requirements.',
        '287.0': 'Confer with creative, art, copywriting, or production department heads to discuss client requirements and presentation concepts and to coordinate creative activities.',
        '297.0': 'Conceptualize and help design interfaces for multimedia games, products, and devices.',
        '3007.0': 'Schedule maintenance for industrial machines and equipment, and keep equipment service records.',
        '301.0': 'Use computer software to generate new images.',
        '303.0': 'Draw and print charts, graphs, illustrations, and other artwork, using computer.',
        '304.0': 'Review final layouts and suggest improvements, as needed.',
        '306.0': 'Develop graphics and layouts for product illustrations, company logos, and Web sites.',
        '3155.0': 'Listen to and resolve customers\' complaints regarding products or services.',
        '3214.0': 'Record product, packaging, and order information on specified forms and records.',
        '3267.0': 'Direct preparation and distribution of written and verbal information to inform employees of benefits, compensation, and personnel policies.',
        '3268.0': 'Administer, direct, and review employee benefit programs, including the integration of benefit programs following mergers and acquisitions.',
        '3277.0': 'Maintain records and compile statistical reports concerning personnel-related data, such as hires, transfers, performance appraisals, and absenteeism rates.',
        '3314.0': 'Confer with supervisory personnel, owners, contractors, or design professionals to discuss and resolve matters, such as work procedures, complaints, or construction problems.',
        '3315.0': 'Plan, organize, or direct activities concerned with the construction or maintenance of structures, facilities, or systems.',
        '3325.0': 'Requisition supplies or materials to complete construction projects.',
        '333.0': 'Write research reports and other publications to document and communicate research findings.',
        '3409.0': 'Analyze applicants\' financial status, credit, and property evaluations to determine feasibility of granting loans.',
        '3410.0': 'Explain to customers the different types of loans and credit options that are available, as well as the terms of those services.',
        '3414.0': 'Compute payment schedules.',
        '3415.0': 'Stay abreast of new types of loans and other financial services and products to better meet customers\' needs.',
        '3465.0': 'Test, maintain, and monitor computer programs and systems, including coordinating the installation of computer programs and systems.',
        '3466.0': 'Use object-oriented programming languages, as well as client and server applications development processes and multimedia and Internet technology.',
        '3467.0': 'Confer with clients regarding the nature of the information processing or computation needs a computer program is to address.',
        '3468.0': 'Coordinate and link the computer systems within an organization to increase compatibility so that information can be shared.',
        '3470.0': 'Expand or modify system to serve new purposes or improve work flow.',
        '3474.0': 'Analyze information processing or computation needs and plan and design computer systems, using techniques such as structured analysis, data modeling, and information engineering.',
        '3475.0': 'Assess the usefulness of pre-developed application packages and adapt them to a user environment.',
        '3476.0': 'Define the goals of the system and devise flow charts and diagrams describing logical operational steps of programs.',
        '3477.0': 'Develop, document, and revise system design procedures, test procedures, and quality standards.',
        '3480.0': 'Read manuals, periodicals, and technical reports to learn how to develop programs that meet staff and user requirements.',
        '3515.0': 'Collect information about specific features of the Earth, using aerial photography and other digital remote sensing techniques.',
        '3523.0': 'Build and update digital databases.',
        '3557.0': 'Develop processes to separate components of liquids or gases or generate electrical currents, using controlled chemical processes.',
        '363.0': 'Plan, prepare, or carry out individually designed programs of physical treatment to maintain, improve, or restore physical functioning, alleviate pain, or prevent physical dysfunction in patients.',
        '3686.0': 'Provide visitor services, such as explaining regulations, answering visitor requests, needs and complaints, and providing information about the park and surrounding areas.',
        '3691.0': 'Assist with operations of general facilities, such as visitor centers.',
        '3704.0': 'Investigate the composition, structure, or history of the Earth\'s crust through the collection, examination, measurement, or classification of soils, minerals, rocks, or fossil remains.',
        '3707.0': 'Assess ground or surface water movement to provide advice on issues, such as waste management, route and site selection, or the restoration of contaminated sites.',
        '3740.0': 'Use computers, computer-interfaced equipment, robotics or high-technology industrial applications to perform work duties.',
        '3746.0': 'Conduct standardized biological, microbiological or biochemical tests and laboratory analyses to evaluate the quantity or quality of physical or chemical substances in food or other products.',
        '3774.0': 'Interpret laws, rulings and regulations for individuals and businesses.',
        '3775.0': 'Analyze the probable outcomes of cases, using knowledge of legal precedents.',
        '3783.0': 'Study Constitution, statutes, decisions, regulations, and ordinances of quasi-judicial bodies to determine ramifications for cases.',
        '3967.0': 'Maintain records and files of work and revisions.',
        '3982.0': 'Mix and regulate sound inputs and feeds or coordinate audio feeds with television pictures.',
        '3990.0': 'Compress, digitize, duplicate, and store audio and video data.',
        '3992.0': 'Edit videotapes by erasing and removing portions of programs and adding video or sound as required.',
        '3997.0': 'Record and edit audio material, such as movie soundtracks, using audio recording and editing equipment.',
        '4008.0': 'Control audio equipment to regulate volume and sound quality during radio and television broadcasts.',
        '4084.0': 'Order medical and laboratory supplies and equipment.',
        '4222.0': 'Care for athletic injuries, using physical therapy equipment, techniques, or medication.',
        '4223.0': 'Evaluate athletes\' readiness to play and provide participation clearances when necessary and warranted.',
        '4229.0': 'Develop training programs or routines designed to improve athletic performance.',
        '4242.0': 'Provide patients and families with emotional support and instruction in areas such as caring for infants, preparing healthy meals, living independently, or adapting to disability or illness.',
        '4296.0': 'Start equipment and observe gauges and equipment operation to detect malfunctions and to ensure equipment is operating to prescribed standards.',
        '4538.0': 'Conduct educational activities for school children.',
        '4539.0': 'Escort individuals or groups on cruises, sightseeing tours, or through places of interest, such as industrial establishments, public buildings, or art galleries.',
        '4627.0': 'Maintain records of contacts, accounts, and orders.',
        '4628.0': 'Schedule appointments for sales representatives to meet with prospective customers or for customers to attend sales presentations.',
        '4630.0': 'Operate communication systems, such as telephone, switchboard, intercom, two-way radio, or public address.',
        '4631.0': 'Answer incoming calls, greeting callers, providing information, transferring calls or taking messages as necessary.',
        '4635.0': 'Place telephone calls or arrange conference calls as instructed.',
        '4639.0': 'Keep records of calls placed and charges incurred.',
        '4722.0': 'Answer customers\' questions and explain available services, such as deposit accounts, bonds, and securities.',
        '4724.0': 'Refer customers to appropriate bank personnel to meet their financial needs.',
        '4726.0': 'Inform customers of procedures for applying for services, such as ATM cards, direct deposit of checks, and certificates of deposit.',
        '4732.0': 'Duplicate records for distribution to branch offices.',
        '4947.0': 'Observe gauges, dials, and product characteristics, and adjust controls to maintain appropriate temperature, pressure, and flow of ingredients.',
        '4950.0': 'Set temperature, pressure, and time controls, and start conveyers, machines, or pumps.',
        '4951.0': 'Tend or operate and control equipment, such as kettles, cookers, vats and tanks, and boilers, to cook ingredients or prepare products for further processing.',
        '4997.0': 'Rinse objects and place them on drying racks or use cloth, squeegees, or air compressors to dry surfaces.',
        '5089.0': 'Transfer animals between enclosures to facilitate breeding, birthing, shipping, or rearrangement of exhibits.',
        '5116.0': 'Provide information about facilities, entertainment options, and rules and regulations.',
        '5117.0': 'Record details of attendance, sales, receipts, reservations, or repair activities.',
        '5133.0': 'Schedule the use of recreation facilities, such as golf courses, tennis courts, bowling alleys, or softball diamonds.',
        '5199.0': 'Direct and coordinate activities of teachers or administrators at daycare centers, schools, public agencies, or institutions.',
        '5200.0': 'Plan, direct, and monitor instructional methods and content of educational, vocational, or student activity programs.',
        '5203.0': 'Review and evaluate new and current programs to determine their efficiency, effectiveness, and compliance with state, local, and federal regulations and recommend any necessary modifications.',
        '5207.0': 'Collect and analyze survey data, regulatory information, and demographic and employment trends to forecast enrollment patterns and the need for curriculum changes.',
        '5208.0': 'Inform businesses, community groups, and governmental agencies about educational needs, available programs, and program policies.',
        '5221.0': 'Direct and coordinate activities of teachers, administrators, and support staff at schools, public agencies, and institutions.',
        '5243.0': 'Direct activities of administrative departments, such as admissions, registration, and career services.',
        '5249.0': 'Participate in student recruitment, selection, and admission, making admissions recommendations when required to do so.',
        '5299.0': 'Examine accounting systems and records to determine whether accounting methods used were appropriate and in compliance with statutory provisions.',
        '5317.0': 'Modify computer security files to incorporate new software, correct errors, or change individual access status.',
        '5319.0': 'Monitor use of data files and regulate access to safeguard information in computer files.',
        '5321.0': 'Encrypt data transmissions and erect firewalls to conceal confidential information as it is being transmitted and to keep out tainted digital transfers.',
        '5339.0': 'Design electronic components, software, products, or systems for commercial, industrial, medical, military, or scientific applications.',
        '5433.0': 'Collect and analyze data on customer demographics, preferences, needs, and buying habits to identify potential markets and factors affecting product demand.',
        '5436.0': 'Forecast and track marketing and sales trends, analyzing collected data.',
        '5439.0': 'Conduct research on consumer opinions and marketing strategies, collaborating with marketing professionals, statisticians, pollsters, and other professionals.',
        '5441.0': 'Gather data on competitors and analyze their prices, sales, and method of marketing and distribution.',
        '5461.0': 'Collect and analyze data to evaluate the effectiveness of academic programs and other services, such as behavioral management systems.',
        '5467.0': 'Collect data about the attitudes, values, and behaviors of people in groups, using observation, interviews, and review of documents.',
        '5509.0': 'Gather and compile geographic data from sources such as censuses, field observations, satellite imagery, aerial photographs, and existing maps.',
        '5511.0': 'Study the economic, political, and cultural characteristics of a specific region\'s population.',
        '5518.0': 'Develop and test theories, using information from interviews, newspapers, periodicals, case law, historical papers, polls, or statistical sources.',
        '5606.0': 'Maintain accurate and complete student records as required by laws, district policies, and administrative regulations.',
        '5616.0': 'Provide students with information on topics such as college degree programs and admission requirements, financial aid opportunities, trade and technical schools, and apprenticeship programs.',
        '5637.0': 'Establish and supervise peer-counseling and peer-tutoring programs.',
        '5664.0': 'Compile, administer, and grade examinations, or assign this work to others.',
        '5665.0': 'Prepare course materials, such as syllabi, homework assignments, and handouts.',
        '5666.0': 'Maintain student attendance records, grades, and other required records.',
        '5667.0': 'Initiate, facilitate, and moderate classroom discussions.',
        '5672.0': 'Select and obtain materials and supplies, such as textbooks.',
        '5673.0': 'Collaborate with colleagues to address teaching and research issues.',
        '5675.0': 'Participate in student recruitment, registration, and placement activities.',
        '5676.0': 'Serve on academic or administrative committees that deal with institutional policies, departmental matters, and academic issues.',
        '5677.0': 'Participate in campus and community events.',
        '5678.0': 'Compile bibliographies of specialized materials for outside reading assignments.',
        '5679.0': 'Perform administrative duties, such as serving as department head.',
        '5682.0': 'Act as advisers to student organizations.',
        '5684.0': 'Write grant proposals to procure external research funding.',
        '5711.0': 'Prepare and deliver lectures to undergraduate or graduate students on topics such as linear algebra, differential equations, and discrete mathematics.',
        '5720.0': 'Collaborate with colleagues to address teaching and research issues.',
        '5757.0': 'Conduct research in a particular field of knowledge and publish findings in professional journals, books, or electronic media.',
        '5767.0': 'Collaborate with colleagues to address teaching and research issues.',
        '5777.0': 'Evaluate and grade students\' class work, laboratory work, assignments, and papers.',
        '5982.0': 'Perform administrative duties, such as serving as department head.',
        '6329.0': 'Perform administrative duties, such as serving as department head.',
        '6340.0': 'Prepare and deliver lectures to undergraduate or graduate students on topics such as how to speak and write a foreign language and the cultural aspects of areas where a particular language is used.',
        '6343.0': 'Keep abreast of developments in their field by reading current literature, talking with colleagues, and participating in professional organizations and activities.',
        '6457.0': 'Develop teaching aids, such as instructional software, multimedia visual aids, or study materials.',
        '6458.0': 'Select and assemble books, materials, supplies, and equipment for training, courses, or projects.',
        '6460.0': 'Participate in conferences, seminars, and training sessions to keep abreast of developments in the field, and integrate relevant information into training programs.',
        '6462.0': 'Review enrollment applications and correspond with applicants to obtain additional information.',
        '6493.0': 'Select, store, order, issue, and inventory classroom equipment, materials, and supplies.',
        '6542.0': 'Assign and grade class work and homework.',
        '6765.0': 'Prepare objectives and outlines for courses of study, following curriculum guidelines or requirements of states and schools.',
        '678.0': 'Compute charges for merchandise or services and receive payments.',
        '680.0': 'Recommend and provide advice on a wide variety of products and services.',
        '684.0': 'Prepare rental forms, obtaining customer signature and other information, such as required licenses.',
        '6850.0': 'Select, order, and issue books, materials, and supplies for courses or projects.',
        '688.0': 'Reserve items for requested times and keep records of items rented.',
        '6882.0': 'Design complex graphics and animation, using independent judgment, creativity, and computer equipment.',
        '6883.0': 'Create two-dimensional and three-dimensional images depicting objects in motion or illustrating a process, using computer animation or modeling programs.',
        '6884.0': 'Make objects or characters appear lifelike by manipulating light, color, texture, shadow, and transparency, or manipulating static images to give the illusion of motion.',
        '6887.0': 'Script, plan, and create animated narrative sequences under tight deadlines, using computer software and hand drawing techniques.',
        '6888.0': 'Create basic designs, drawings, and illustrations for product labels, cartons, direct mail, or television.',
        '6890.0': 'Develop briefings, brochures, multimedia presentations, web pages, promotional products, technical illustrations, and computer artwork for use in products, technical manuals, literature, newsletters, and slide shows.',
        '6898.0': 'Modify and refine designs, using working models, to conform with customer specifications, production limitations, or changes in design trends.',
        '6902.0': 'Evaluate feasibility of design ideas, based on factors such as appearance, safety, function, serviceability, budget, production costs/methods, and market characteristics.',
        '694.0': 'Greet customers and ascertain what each customer wants or needs.',
        '698.0': 'Maintain records related to sales.',
        '7067.0': 'Fabricate concrete beams, columns, and panels.',
        '7253.0': 'Keep informed of activities or changes that could affect the likelihood of an emergency, response efforts, or plan implementation.',
        '7257.0': 'Coordinate disaster response or crisis management activities, such as ordering evacuations, opening public shelters, and implementing special needs plans and programs.',
        '7258.0': 'Develop and maintain liaisons with municipalities, county departments, and similar entities to facilitate plan development, response effort coordination, and exchanges of personnel and equipment.',
        '7260.0': 'Prepare emergency situation status reports that describe response and recovery efforts, needs, and preliminary damage assessments.',
        '7277.0': 'Analyze data gathered and develop solutions or alternative methods of proceeding.',
        '7314.0': 'Investigate activities of institutions to enforce laws and regulations and to ensure legality of transactions and operations or financial solvency.',
        '7318.0': 'Examine the minutes of meetings of directors, stockholders, and committees to investigate the specific authority extended at various levels of management.',
        '7320.0': 'Review balance sheets, operating income and expense accounts, and loan documentation to confirm institution assets and liabilities.',
        '7326.0': 'Review applications for mergers, acquisitions, establishment of new institutions, acceptance in Federal Reserve System, or registration of securities sales to determine their public interest value and conformance to regulations, and recommend acceptance or rejection.',
        '7367.0': 'Apply mathematical theories and techniques to the solution of practical problems in business, engineering, the sciences, or other fields.',
        '7370.0': 'Perform computations and apply methods of numerical analysis to data.',
        '7371.0': 'Develop mathematical or statistical models of phenomena to be used for analysis or for computational simulation.',
        '7372.0': 'Assemble sets of assumptions, and explore the consequences of each set.',
        '7374.0': 'Develop new principles and new relationships between existing mathematical principles to advance mathematical science.',
        '7375.0': 'Design, analyze, and decipher encryption systems designed to transmit military, political, financial, or law-enforcement-related information in code.',
        '7376.0': 'Conduct research to extend mathematical knowledge in traditional areas, such as algebra, geometry, probability, and logic.',
        '7384.0': 'Prepare management reports defining and evaluating problems and recommending solutions.',
        '739.0': 'Compute cost of travel and accommodations, using calculator, computer, carrier tariff books, and hotel rate books, or quote package tour\'s costs.',
        '7393.0': 'Test and verify hardware and support peripherals to ensure that they meet specifications and requirements, by recording and analyzing test data.',
        '7394.0': 'Monitor functioning of equipment and make necessary modifications to ensure system operates in conformance with specifications.',
        '7399.0': 'Confer with engineering staff and consult specifications to evaluate interface between hardware and software and operational and performance requirements of overall system.',
        '740.0': 'Book transportation and hotel reservations, using computer or telephone.',
        '741.0': 'Plan, describe, arrange, and sell itinerary tour packages and promotional travel incentives offered by various travel carriers.',
        '743.0': 'Print or request transportation carrier tickets, using computer printer system or system link to travel carrier.',
        '745.0': 'Receive payment and record receipts for services.',
        '7485.0': 'Test new products for flavor, texture, color, nutritional content, and adherence to government and industry standards.',
        '7489.0': 'Study methods to improve aspects of foods, such as chemical composition, flavor, color, texture, nutritional value, and convenience.',
        '7497.0': 'Measure and assess vegetation resources for biological assessment companies, environmental impact statements, and rangeland monitoring programs.',
        '7524.0': 'Study celestial phenomena, using a variety of ground-based and space-borne telescopes and scientific instruments.',
        '7525.0': 'Analyze research data to determine its significance, using computers.',
        '7527.0': 'Measure radio, infrared, gamma, and x-ray emissions from extraterrestrial sources.',
        '7531.0': 'Develop instrumentation and software for astronomical observation and analysis.',
        '7534.0': 'Calculate orbits and determine sizes, shapes, brightness, and motions of different celestial bodies.',
        '7538.0': 'Compile, analyze, and report data to explain economic phenomena and forecast market trends, applying mathematical models and statistical techniques.',
        '754.0': 'Keep a current record of staff members\' whereabouts and availability.',
        '7550.0': 'Conduct surveys and collect data, using methods such as interviews, questionnaires, focus groups, market analysis surveys, public opinion polls, literature reviews, and file reviews.',
        '7576.0': 'Conduct standardized tests on food, beverages, additives, or preservatives to ensure compliance with standards and regulations regarding factors such as color, texture, or nutrients.',
        '759.0': 'Schedule space or equipment for special programs and prepare lists of participants.',
        '7630.0': 'Create and maintain accessible, retrievable computer archives and databases, incorporating current advances in electronic information storage technology.',
        '7631.0': 'Organize archival records and develop classification systems to facilitate access to archival materials.',
        '7636.0': 'Preserve records, documents, and objects, copying records to film, videotape, audiotape, disk, or computer formats as necessary.',
        '7639.0': 'Research and record the origins and historical significance of archival materials.',
        '7654.0': 'Write original or adapted material for dramas, comedies, puppet shows, narration, or other performances.',
        '772.0': 'Review legal publications and perform database searches to identify laws and court decisions relevant to pending cases.',
        '7727.0': 'Record speech, music, and other sounds on recording media, using recording equipment.',
        '7729.0': 'Separate instruments, vocals, and other sounds, and combine sounds during the mixing or postproduction stage.',
        '7731.0': 'Create musical instrument digital interface programs for music projects, commercials, or film postproduction.',
        '775.0': 'Schedule and confirm patient diagnostic appointments, surgeries, or medical consultations.',
        '7769.0': 'Instruct individuals and groups on ways to preserve health and prevent disease.',
        '7774.0': 'Collect, record, and maintain patient information, such as medical history, reports, or examination results.',
        '7799.0': 'Provide consulting services to other doctors caring for patients with special or difficult problems.',
        '7856.0': 'Perform administrative duties, such as hiring employees, ordering supplies, or keeping records.',
        '7941.0': 'Identify vehicles in violation of parking codes, checking with dispatchers when necessary to confirm identities or to determine whether vehicles need to be booted or towed.',
        '796.0': 'Keep records of work performed.',
        '8122.0': 'Monitor market conditions, product innovations, and competitors\' products, prices, and sales.',
        '8150.0': 'Monitor daily stock prices and compute fluctuations to determine the need for additional collateral to secure loans.',
        '8190.0': 'Advise clients on transportation and payment methods.',
        '8452.0': 'Read and analyze charts, work orders, production schedules, and other records and reports to determine production requirements and to evaluate current production estimates and outputs.',
        '8454.0': 'Plan and establish work schedules, assignments, and production sequences to meet production goals.',
        '8457.0': 'Observe work and monitor gauges, dials, and other indicators to ensure that operators conform to production or processing standards.',
        '8460.0': 'Maintain operations data, such as time, production, and cost records, and prepare management reports of production results.',
        '8616.0': 'Inspect and maintain vehicle supplies and equipment, such as gas, oil, water, tires, lights, or brakes, to ensure that vehicles are in proper working condition.',
        '8617.0': 'Report any mechanical problems encountered with vehicles.',
        '8677.0': 'Participate in publicity planning and student recruitment.',
        '8842.0': 'Nominate citizens to boards or commissions.',
        '8849.0': 'Attend and participate in meetings of municipal councils or council committees.',
        '8856.0': 'Prepare or direct preparation of financial statements, business activity reports, financial position forecasts, annual budgets, or reports required by regulatory agencies.',
        '8868.0': 'Advise management on short-term and long-term financial objectives, policies, and actions.',
        '8932.0': 'Maintain and develop positive business relationships with a customer\'s key personnel involved in, or directly relevant to, a logistics activity.',
        '8936.0': 'Protect and control proprietary materials.',
        '8937.0': 'Review logistics performance with customers against targets, benchmarks, and service agreements.',
        '8953.0': 'Report results of statistical analyses, including information in the form of graphs, charts, and tables.',
        '8954.0': 'Process large amounts of data for statistical modeling and graphic analysis, using computers.',
        '8956.0': 'Analyze and interpret statistical data to identify significant differences in relationships among sources of information.',
        '8958.0': 'Evaluate the statistical methods and procedures used to obtain data to ensure validity, applicability, efficiency, and accuracy.',
        '8965.0': 'Adapt statistical methods to solve specific problems in many fields, such as economics, biology, and engineering.',
        '8973.0': 'Develop models or computer simulations of human biobehavioral systems to obtain data for measuring or controlling life processes.',
        '9068.0': 'Broadcast weather conditions, forecasts, or severe weather warnings to the public via television, radio, or the Internet or provide this information to the news media.',
        '9069.0': 'Gather data from sources such as surface or upper air stations, satellites, weather bureaus, or radar for use in meteorological reports or forecasts.',
        '9232.0': 'Read search requests to ascertain types of title evidence required and to obtain descriptions of properties and names of involved parties.',
        '9240.0': 'Obtain maps or drawings delineating properties from company title plants, county surveyors, or assessors\' offices.',
        '9247.0': 'Create functional or decorative objects by hand, using a variety of methods and materials.',
        '9252.0': 'Develop concepts or creative ideas for craft objects.',
        '9254.0': 'Confer with customers to assess customer needs or obtain feedback.',
        '9261.0': 'Develop designs using specialized computer software.',
        '93.0': 'Keep up with developments in area of expertise by reading current journals, books, or magazine articles.',
        '9328.0': 'Translate messages simultaneously or consecutively into specified languages, orally or by using hand signs, maintaining message content, context, and style as much as possible.',
        '9330.0': 'Check translations of technical terms and terminology to ensure that they are accurate and remain consistent throughout translation revisions.',
        '9331.0': 'Read written materials, such as legal documents, scientific works, or news reports, and rewrite material into specified languages.',
        '9332.0': 'Refer to reference materials, such as dictionaries, lexicons, encyclopedias, and computerized terminology banks, as needed to ensure translation accuracy.',
        '9344.0': 'Adjust apertures, shutter speeds, and camera focus according to a combination of factors, such as lighting, field depth, subject motion, film type, and film speed.',
        '9345.0': 'Use traditional or digital cameras, along with a variety of equipment, such as tripods, filters, and flash attachments.',
        '9352.0': 'Manipulate and enhance scanned or digital images to create desired effects, using computers and specialized software.',
        '9367.0': 'License the use of photographs through stock photo agencies.',
        '9450.0': 'Examine immigration applications, visas, and passports and interview persons to determine eligibility for admission, residence, and travel in the U.S.',
        '9519.0': 'Inspect and evaluate the physical condition of facilities to determine the type of work required.',
        '9528.0': 'Recommend or arrange for additional services, such as painting, repair work, renovations, and the replacement of furnishings and equipment.',
        '9623.0': 'Check the appearance of costumes on stage or under lights to determine whether desired effects are being achieved.',
        '966.0': 'Manage backup, security and user help systems.',
        '9660.0': 'Listen to and resolve customer complaints regarding services, products, or personnel.',
        '967.0': 'Consult with users, management, vendors, and technicians to assess computing needs and system requirements.',
        '968.0': 'Direct daily operations of department, analyzing workflow, establishing priorities, developing standards and setting deadlines.',
        '970.0': 'Stay abreast of advances in technology.',
        '971.0': 'Develop computer information resources, providing for data security and control, strategic computing, and disaster recovery.',
        '972.0': 'Review and approve all systems charts and programs prior to their implementation.',
        '973.0': 'Evaluate the organization\'s technology use and needs and recommend improvements, such as hardware and software upgrades.',
        '9749.0': 'Plan routes, itineraries, and accommodation details, and compute fares and fees, using schedules, rate books, and computers.',
        '975.0': 'Meet with department heads, managers, supervisors, vendors, and others, to solicit cooperation and resolve problems.',
        '9750.0': 'Make and confirm reservations for transportation and accommodations, using telephones, faxes, mail, and computers.',
        '9752.0': 'Answer inquiries regarding information, such as schedules, accommodations, procedures, or policies.',
        '9754.0': 'Determine whether space is available on travel dates requested by customers, assigning requested spaces when available.',
        '9755.0': 'Inform clients of essential travel information, such as travel times, transportation connections, or medical and visa requirements.',
        '9756.0': 'Maintain computerized inventories of available passenger space and provide information on space reserved or available.',
        '9761.0': 'Announce arrival and departure information, using public address systems.',
        '9762.0': 'Trace lost, delayed, or misdirected baggage for customers.',
        '9766.0': 'Provide customers with travel suggestions and information sources, such as guides, directories, brochures, or maps.',
        '977.0': 'Recruit, hire, train and supervise staff, or participate in staffing decisions.',
        '9788.0': 'Compute and analyze data, using statistical formulas and computers or calculators.',
        '9790.0': 'Compile statistics from source materials, such as production or sales records, quality-control or test records, time sheets, or survey sheets.',
        '9793.0': 'Participate in the publication of data or information.',
        '9795.0': 'File data and related information, and maintain and update databases.',
        '983.0': 'Identify staff vacancies and recruit, interview, and select applicants.',
        '984.0': 'Allocate human resources, ensuring appropriate matches between personnel.',
        '985.0': 'Provide current and prospective employees with information about policies, job duties, working conditions, wages, opportunities for promotion, and employee benefits.',
        '986.0': 'Perform difficult staffing duties, including dealing with understaffing, refereeing disputes, firing employees, and administering disciplinary procedures.',
        '990.0': 'Serve as a link between management and employees by handling questions, interpreting and administering contracts and helping resolve work-related problems.',
        '991.0': 'Plan, direct, supervise, and coordinate work activities of subordinates and staff relating to employment, compensation, labor relations, and employee relations.',
        '993.0': 'Maintain records and compile statistical reports concerning personnel-related data such as hires, transfers, performance appraisals, and absenteeism rates.',
        '994.0': 'Analyze statistical data and reports to identify and determine causes of personnel problems and develop recommendations for improvement of organization\'s personnel policies and practices.',
    }
};

// Get L2 clusters for a given L1 cluster
function getL2ClustersForL1(l1Id) {
    var l2Clusters = [];
    for (var l2Id in onetData.l2ToL1) {
        if (onetData.l2ToL1[l2Id] === l1Id) {
            l2Clusters.push({
                value: l2Id,
                text: onetData.l2Clusters[l2Id]
            });
        }
    }
    return l2Clusters;
}

// Get tasks for a given L2 cluster
function getTasksForL2(l2Id) {
    var tasks = [];
    for (var taskId in onetData.taskToL2) {
        if (onetData.taskToL2[taskId] === l2Id) {
            tasks.push({
                value: taskId,
                text: onetData.tasks[taskId]
            });
        }
    }
    return tasks;
}

// Functionality Hierarchy Data Structures
var functionalityData = {
    mainCategories: {
        'perception': 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data',
        'reasoning': 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns',
        'action': 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files',
    },

    subCategories: {
        'perception': [
            { value: 'sensors', text: 'Sensors - e.g., internal database, monitoring, diagnostics, GUI, voice, internet search, physical world' },
        ],
        'reasoning': [
            { value: 'planning', text: 'Planning - e.g., task-decomposition, path-finding models' },
            { value: 'analysis', text: 'Analysis - e.g., scratchpads, calculators, simulations' },
            { value: 'resource_management', text: 'Resource management - e.g., memory, self-management' },
            { value: 'unclear', text: 'Unclear - does not fit into the above categories' },
        ],
        'action': [
            { value: 'authentication', text: 'Authentication - e.g., login, CAPTCHA, wallet' },
            { value: 'computer_use', text: 'Computer use - e.g., application-specific GUI interaction, website interactions, computer use' },
            { value: 'code_execution', text: 'Running code - e.g., sandboxed code interpreter, IDE, file operations, code execution' },
            { value: 'software_extensions', text: 'Software extensions - e.g., calendar, social media API' },
            { value: 'physical_extensions', text: 'Physical extensions - e.g., robotic arm, laboratory tools in factory setting, robot in an open environment' },
            { value: 'human_interaction', text: 'Human interaction - e.g., phone calls' },
            { value: 'agent_interaction', text: 'Agent interaction - e.g., multi-agent simulation, sub-agents that can interact with outside world, third-party agent interactions' },
            { value: 'unclear', text: 'Unclear - does not fit into the above categories' },
        ],
    }
};

// Get sub-categories for a given main category
function getSubCategoriesFor(mainCategory) {
    return functionalityData.subCategories[mainCategory] || [];
}

// Study data - pages with hierarchical validation questions
var studyPages = [
    {
        type: 'instructions',
        title: 'Study Instructions',
        content: `# How to Classify MCP Servers

Thank you for participating! You will be evaluating MCP (Model Context Protocol) servers - tools that give AI agents specific capabilities.

## Your Task

For each server, you will:

* **Classify occupational function** (Q1.1) - What job task does it support? (Note: Many servers will be in the category 'Design, implement, and maintain diverse information technology systems')

* **Classify functionality type** (Q2.1-Q2.2) - How does it work?

* **Answer standard questions** (Q3-Q5) - Industry specificity, environment, and payment autonomy
* Based on our testing, the first servers will take you 2-10mins until you get used to it, then we expect each server to take around 1-1.5mins.
* Click **Next** to see example classifications, then you'll practice with feedback before starting the real study.`
    },
    // Tutorial Example 1: Pre-answered (asher-mcp)
    {
        type: 'tutorial_intro',
        exampleNumber: 1,
        title: 'Tutorial Example 1/2 (Pre-answered)',
        content: `# Example 1: Pre-answered Classification

We'll show you a complete classification with all correct answers. Study this carefully to understand the reasoning.

* **Server**: asher-mcp

* **Description**: Financial data aggregation tool that connects to banking APIs to retrieve account information, balances, transactions, and investment holdings

* **Tools (sample)**:
* get_accounts: Retrieve list of all connected bank accounts
* get_account_balance: Get current balance for a specific account
* get_transactions: Retrieve transaction history for an account
* get_investment_holdings: View investment portfolio holdings

All these tools are **read-only** - they retrieve information but don't execute any actions.

Click **Next** to see the correct classifications.`
    },
    {
        type: 'tutorial_preanswered',
        exampleNumber: 1,
        title: 'Tutorial Example 1/2 - Correct Answers',
        serverName: 'asher-mcp',
        description: 'Financial data aggregation tool that connects to banking APIs to retrieve account information, balances, transactions, and investment holdings',
        tools: [
            'get_accounts: Retrieve list of all connected bank accounts',
            'get_account_balance: Get current balance for a specific account',
            'get_transactions: Retrieve transaction history for an account',
            'get_investment_holdings: View investment portfolio holdings'
        ],
        correctAnswers: {
            onet_l1: '13',
            onet_l1_text: 'Business and Financial Operations',
            onet_l1_explanation: `- PRIMARY function: retrieving and monitoring financial account information
- Maps directly to financial operations tasks
- Not general software development (that would be Computer/Mathematical)`,

            onet_l2: '13-2',
            onet_l2_text: 'Financial Specialists',
            onet_l2_explanation: `- Specifically handles financial data aggregation and account monitoring
- Aligns with financial specialist roles (not general business operations)
- Requires understanding of financial systems and banking APIs`,

            onet_task: '4.A.2.b.4',
            onet_task_text: 'Monitor financial data and prepare financial reports',
            onet_task_explanation: `- Core function is monitoring account balances, transactions, and holdings
- Provides data that feeds into financial reporting
- Read-only monitoring rather than active trading or execution`,

            func_main: 'perception',
            func_main_text: 'Perception (gathering information)',
            func_main_explanation: `- All tools are read-only data retrieval (no calculations, no actions)
- "Perception" = gathering information from external systems
- NOT "Action" because nothing is modified or executed`,

            func_sub: 'sensors',
            func_sub_text: 'Sensors',
            func_sub_explanation: `- Acts as sensors connecting to banking APIs
- Retrieves data without processing or transforming it
- No autonomous execution - just data collection`,

            q3: '0',
            q3_text: 'Industry-specific (finance)',
            q3_explanation: `- Only useful for finance sector
- Tools specifically designed for banking/financial accounts
- Not cross-industry like general file management tools`,

            q4: '0',
            q4_text: 'Trusted/pre-specified (specific banking APIs)',
            q4_explanation: `- Uses specific banking APIs (Plaid, Yodlee, etc.)
- Not open web scraping or generic file access
- Requires authentication to specific financial institutions`,

            q5: '1',
            q5_text: 'Information only (read-only payment data)',
            q5_explanation: `- Can VIEW payment data and transaction history
- Cannot initiate, generate, or execute payments
- Read-only access to financial information`
        }
    },
    // Tutorial Example 2: Practice intro (base-mcp)
    {
        type: 'tutorial_intro',
        exampleNumber: 2,
        title: 'Tutorial Example 2/2 (Practice)',
        content: `# Example 2: Practice Classification

Now it's your turn! Classify this server just as you would in the real study.

**Server**: base-mcp

**Description**: Blockchain interaction tool for Base network (Ethereum L2). Enables checking wallet balances, sending ETH/tokens, and deploying/interacting with smart contracts.

**Tools (sample)**:
* get_balance: Check wallet balance for ETH and tokens
* send_transaction: Send ETH or tokens to another address
* deploy_contract: Deploy new smart contracts to the blockchain
* call_contract: Execute functions on existing smart contracts

Notice the key difference from Example 1: this server can **EXECUTE** transactions, not just read data.

Click **Next** to start classifying (you'll answer all 6 questions).`
    },
    // Tutorial Practice 2: Q1.1 - O*NET Level 1
    {
        type: 'onet_l1',
        serverIndex: 'tutorial_2',
        title: 'Tutorial Example 2/2 - Practice Question 1/6',
        serverName: 'base-mcp',
        serverId: 'tutorial_2',
        description: 'Blockchain interaction tool for Base network (Ethereum L2). Enables checking wallet balances, sending ETH/tokens, and deploying/interacting with smart contracts.',
        tools: [
            'get_balance: Check wallet balance for ETH and tokens',
            'send_transaction: Send ETH or tokens to another address',
            'deploy_contract: Deploy new smart contracts to the blockchain',
            'call_contract: Execute functions on existing smart contracts'
        ],
        question: 'onet_l1',
        questionText: 'Q1.1: Which broad occupational category best describes the primary function of this server?',
        isPractice: true,
        correctAnswer: 'L1_01'
    },
    // Tutorial Practice 2: Q1.1 Feedback
    {
        type: 'tutorial_feedback',
        serverIndex: 'tutorial_2',
        title: 'Tutorial Example 2/2 - Feedback for Q1.1',
        questionKey: 'onet_l1',
        questionTitle: 'O*NET Level 1 - Broad Category',
        correctValue: 'L1_01',
        feedbackTip: 'Focus on the PRIMARY occupational function. What job role would most commonly use this server? If it\'s for general software tasks, choose Computer/Mathematical. If it\'s for business operations, choose Business Operations.',
        isPractice: true
    },
    // Tutorial Practice 2: Q2.1 - Functionality Main
    {
        type: 'func_main',
        serverIndex: 'tutorial_2',
        title: 'Tutorial Example 2/2 - Practice Question 2/6',
        serverName: 'base-mcp',
        serverId: 'tutorial_2',
        description: 'Blockchain interaction tool for Base network (Ethereum L2). Enables checking wallet balances, sending ETH/tokens, and deploying/interacting with smart contracts.',
        tools: [
            'get_balance: Check wallet balance for ETH and tokens',
            'send_transaction: Send ETH or tokens to another address',
            'deploy_contract: Deploy new smart contracts to the blockchain',
            'call_contract: Execute functions on existing smart contracts'
        ],
        question: 'func_main',
        questionText: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        isPractice: true,
        correctAnswer: 'action'
    },
    // Tutorial Practice 2: Q2.1 Feedback
    {
        type: 'tutorial_feedback',
        serverIndex: 'tutorial_2',
        title: 'Tutorial Example 2/2 - Feedback for Q2.1',
        questionKey: 'func_main',
        questionTitle: 'Functionality Main - Primary Type',
        correctValue: 'action',
        feedbackTip: 'Perception = gathering data, Reasoning = processing/analyzing, Action = modifying/executing. Choose based on what the server PRIMARILY does.',
        isPractice: true
    },
    // Tutorial Practice 2: Q2.2 - Functionality Sub
    {
        type: 'func_sub',
        serverIndex: 'tutorial_2',
        title: 'Tutorial Example 2/2 - Practice Question 3/6',
        serverName: 'base-mcp',
        serverId: 'tutorial_2',
        description: 'Blockchain interaction tool for Base network (Ethereum L2). Enables checking wallet balances, sending ETH/tokens, and deploying/interacting with smart contracts.',
        tools: [
            'get_balance: Check wallet balance for ETH and tokens',
            'send_transaction: Send ETH or tokens to another address',
            'deploy_contract: Deploy new smart contracts to the blockchain',
            'call_contract: Execute functions on existing smart contracts'
        ],
        question: 'func_sub',
        questionText: 'Q2.2: Which specific sub-category best describes this server\'s functionality?',
        conditionalOn: 'func_main',
        isPractice: true,
        correctAnswer: 'software_extensions'
    },
    // Tutorial Practice 2: Q2.2 Feedback
    {
        type: 'tutorial_feedback',
        serverIndex: 'tutorial_2',
        title: 'Tutorial Example 2/2 - Feedback for Q2.2',
        questionKey: 'func_sub',
        questionTitle: 'Functionality Sub - Specific Mechanism',
        correctValue: 'software_extensions',
        feedbackTip: 'Within your main category, identify the specific implementation mechanism. Consider how the server technically accomplishes its function.',
        isPractice: true
    },
    // Tutorial Practice 2: Question 4/6 - q3
    {
        type: 'server',
        serverIndex: 'tutorial_2',
        title: 'Tutorial Example 2/2 - Practice Question 4/6',
        serverName: 'base-mcp',
        serverId: 'tutorial_2',
        description: 'Blockchain interaction tool for Base network (Ethereum L2). Enables checking wallet balances, sending ETH/tokens, and deploying/interacting with smart contracts.',
        tools: [
            'get_balance: Check wallet balance for ETH and tokens',
            'send_transaction: Send ETH or tokens to another address',
            'deploy_contract: Deploy new smart contracts to the blockchain',
            'call_contract: Execute functions on existing smart contracts'
        ],
        question: 'q3',
        questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
        choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ],
        isPractice: true,
        correctAnswer: '0'
    },
    // Tutorial Practice 2: q3 Feedback
    {
        type: 'tutorial_feedback',
        serverIndex: 'tutorial_2',
        title: 'Tutorial Example 2/2 - Feedback for Q1',
        questionKey: 'q3',
        questionTitle: 'Industry Generality',
        correctValue: '0',
        feedbackTip: 'Cross-industry tools work across many sectors (like file managers). Industry-specific tools are designed for particular domains (like medical, finance, crypto).',
        isPractice: true
    },
    // Tutorial Practice 2: Question 5/6 - q4
    {
        type: 'server',
        serverIndex: 'tutorial_2',
        title: 'Tutorial Example 2/2 - Practice Question 5/6',
        serverName: 'base-mcp',
        serverId: 'tutorial_2',
        description: 'Blockchain interaction tool for Base network (Ethereum L2). Enables checking wallet balances, sending ETH/tokens, and deploying/interacting with smart contracts.',
        tools: [
            'get_balance: Check wallet balance for ETH and tokens',
            'send_transaction: Send ETH or tokens to another address',
            'deploy_contract: Deploy new smart contracts to the blockchain',
            'call_contract: Execute functions on existing smart contracts'
        ],
        question: 'q4',
        questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
        choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ],
        isPractice: true,
        correctAnswer: '0'
    },
    // Tutorial Practice 2: q4 Feedback
    {
        type: 'tutorial_feedback',
        serverIndex: 'tutorial_2',
        title: 'Tutorial Example 2/2 - Feedback for Q2',
        questionKey: 'q4',
        questionTitle: 'Environment Generality',
        correctValue: '0',
        feedbackTip: 'Open/untrusted means generic internet/file access. Trusted means specific pre-configured APIs or internal systems.',
        isPractice: true
    },
    // Tutorial Practice 2: Question 6/6 - q5
    {
        type: 'server',
        serverIndex: 'tutorial_2',
        title: 'Tutorial Example 2/2 - Practice Question 6/6',
        serverName: 'base-mcp',
        serverId: 'tutorial_2',
        description: 'Blockchain interaction tool for Base network (Ethereum L2). Enables checking wallet balances, sending ETH/tokens, and deploying/interacting with smart contracts.',
        tools: [
            'get_balance: Check wallet balance for ETH and tokens',
            'send_transaction: Send ETH or tokens to another address',
            'deploy_contract: Deploy new smart contracts to the blockchain',
            'call_contract: Execute functions on existing smart contracts'
        ],
        question: 'q5',
        questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
        choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ],
        isPractice: true,
        correctAnswer: '4'
    },
    // Tutorial Practice 2: q5 Feedback
    {
        type: 'tutorial_feedback',
        serverIndex: 'tutorial_2',
        title: 'Tutorial Example 2/2 - Feedback for Q3',
        questionKey: 'q5',
        questionTitle: 'Payment Autonomy Level',
        correctValue: '4',
        feedbackTip: 'Level 0 = no payment capability. Level 1 = read payment data. Level 2 = payment requests or links. Level 3 = third-party processing (e.g., Stripe, PayPal). Level 4 = direct execution (e.g., crypto, credit cards).',
        isPractice: true
    },
    // Server 1 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 1,
        title: 'Server 1/100',
        serverName: 'Obsidian Omnisearch',
        serverId: '@anpigon/mcp-server-obsidian-omnisearch',
        description: 'Search through your Obsidian vault notes programmatically with ease. Integrate seamlessly with other services using a REST API to retrieve absolute paths to matching notes. Enhance your note-taking experience by leveraging powerful search capabilities.',
        tools: [
            'obsidian_notes_search: Search Obsidian(옵시디언) notes and return absolute paths to the matching notes.\n        The returned paths can be used with the...',
            'read_note: Read and return the contents of an Obsidian note file.'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 2 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 2,
        title: 'Server 2/100',
        serverName: 'atlassian-mcp-server',
        serverId: 'rogervinas/atlassian-mcp-server',
        description: '🤖 Simple MCP Server for Atlassian Jira and Confluence',
        tools: [
            'jira_query: Query and retrieve Jira tickets from specified projects, with ability to filter by time period and summarize ticket information. Requir...',
            'confluence_search: Search and retrieve pages from Confluence spaces, with ability to find specific content and summarize page information. Requires...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 3 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 3,
        title: 'Server 3/100',
        serverName: 'snowflake-generic-mcp',
        serverId: 'uniquejtx/snowflake-generic-mcp',
        description: 'General-Purpose Snowflake MCP server for any type of SQL operation',
        tools: [
            'snowflake_authentication: Authenticate to Snowflake using multiple methods: password authentication (SNOWFLAKE_PASSWORD), private key authenticatio...',
            'execute_sql_query: Execute SQL queries on Snowflake data warehouse with proper formatting, metadata, and row counts. Handles query execution with c...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 4 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 4,
        title: 'Server 4/100',
        serverName: 'smartintern-mcp',
        serverId: 'garblesnarff/smartintern-mcp',
        description: 'Multi-agent Slack MCP server with persistent memory - transforms AI into proactive workspace collaborator',
        tools: [
            'slack_send_message: Sends contextual messages to Slack channels or threads based on AI analysis and workspace understanding',
            'slack_manage_threads: Manages Slack conversation threads for organized communication and context tracking',
            'send_action_item_reminders: Sends proactive reminders for pending action items to relevant team members in Slack',
            'analyze_meeting_transcript: Analyzes meeting transcripts to generate structured summaries with extracted action items and key decisions',
            'create_action_item: Creates tracked action items identified from conversations and stores them in the persistent memory system'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 5 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 5,
        title: 'Server 5/100',
        serverName: 'MyMCPSpace ',
        serverId: '@glifxyz/mymcpspace-mcp-server',
        description: 'A social netwok for bots! Interact with your fellow AI agents, no humans allowed',
        tools: [
            'create-post: Create a new post with the provided content',
            'reply-to-post: Create a reply to an existing post',
            'toggle-like: Like or unlike a post',
            'update-username: Update the authenticated user\'s username',
            'get-feed: Get recent posts feed (50 most recent posts in reverse chronological order) along with the current topic'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 6 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 6,
        title: 'Server 6/100',
        serverName: 'mcp-server-generator',
        serverId: 'serhatuzbas/mcp-server-generator',
        description: 'An MCP server that creates and manages  MCP servers! Helps both non-technical users and developers build custom JavaScript MCP servers with AI guidance, automatic dependency management, and Claude Desktop integration.',
        tools: [
            'listServers: List all available servers',
            'getServerContent: View the code of an existing server',
            'createMcpServer: Create a new server',
            'updateMcpServer: Update an existing server',
            'analyzeServerDependenciesinstallServerDependencies: Install required packages'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 7 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 7,
        title: 'Server 7/100',
        serverName: 'uml-mcp',
        serverId: 'antoinebou12/uml-mcp',
        description: 'UML-MCP Server is a UML diagram generation tool based on MCP (Model Context Protocol), which can help users generate various types of UML diagrams through natural language description or directly writing PlantUML and Mermaid and Kroki',
        tools: [
            'generate_uml: Generate any UML diagram. Parameters: diagram_type (type of UML diagram to generate), code (UML diagram code), output_dir (directory ...',
            'generate_class_diagram: Generate UML class diagrams showing classes, attributes, methods, and relationships. Parameters: code (UML class diagram co...',
            'generate_sequence_diagram: Generate UML sequence diagrams showing interactions between objects over time. Parameters: code (UML sequence diagram co...',
            'generate_activity_diagram: Generate UML activity diagrams showing workflows and business processes. Parameters: code (UML activity diagram code), o...',
            'generate_usecase_diagram: Generate UML use case diagrams. Parameters: code (UML use case diagram code), output_dir (directory to save the generated...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 8 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 8,
        title: 'Server 8/100',
        serverName: 'streamersonglist-mcp',
        serverId: 'vuvuvu/streamersonglist-mcp',
        description: 'A Model Context Protocol server for StreamerSongList APIs - works with Claude Desktop and other MCP clients',
        tools: [
            'manageSongRequest: Create, update, and delete song requests. Parameters: action (string, required) - The action to perform ("create", "update", or ...',
            'manageSongAttributes: Add, update, or remove song attributes like tags and ratings. Parameters: action (string, required) - The action to perform (...',
            'getStreamerByName: Fetch detailed information about a specific streamer. Parameters: streamerName (string, required) - The name of the streamer',
            'getQueue: View current song queues with pagination support. Parameters: streamerName (string, required) - The name of the streamer whose queue to f...',
            'getQueueStats: Get comprehensive stats about song queues including total songs, duration, and popular tracks. Parameters: streamerName (string, req...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 9 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 9,
        title: 'Server 9/100',
        serverName: 'simple-memory-mcp-server',
        serverId: 'u1i/simple-memory-mcp-server',
        description: 'Simple MCP server tutorial showing how to build memory tools for LLMs',
        tools: [
            'store_user_info: Store a single piece of information about the user as a one-line fact. Parameters: fact (one line describing something about the u...',
            'get_user_info: Retrieve ALL stored information about the user in a simple numbered list. Parameters: user_id (optional, defaults to \'default\'). Ret...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 10 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 10,
        title: 'Server 10/100',
        serverName: 'mailjet-mcp-server',
        serverId: 'mailgun/mailjet-mcp-server',
        description: 'Official MCP server which allows AI agents to interact with contact, campaign, segmentation, statistics, workflow (and more) APIs from',
        tools: [
            'contact_management: Interact with Mailjet\'s contact API to manage email contacts, including querying and retrieving contact information',
            'campaign_management: Access Mailjet\'s campaign API to create, manage, and monitor email campaigns',
            'segmentation: Use Mailjet\'s segmentation API to organize and categorize contacts into targeted groups',
            'workflow_automation: Interact with Mailjet\'s workflow API to automate email marketing processes and sequences',
            'statistics: Retrieve email campaign statistics and analytics data from Mailjet'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 11 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 11,
        title: 'Server 11/100',
        serverName: 'mcp-yfinance',
        serverId: '9nate-drake/mcp-yfinance',
        description: 'MCP Server for fething yfinance financial data into Claude Desktop',
        tools: [
            'get_stock_price: Retrieves current stock price for a given ticker symbol (e.g., MSFT, AAPL)',
            'get_historical_data: Analyzes historical stock performance over a specified time period (e.g., last 3 months)',
            'get_institutional_ownership: Retrieves institutional ownership information for a stock ticker',
            'get_analyst_targets: Retrieves analyst price targets including median price target for a stock ticker'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 12 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 12,
        title: 'Server 12/100',
        serverName: 'playwright-mcp-server',
        serverId: 'idavidov13/playwright-mcp-server',
        description: 'Set Up and Utilization of Playwright MCP (Model Context Protocol) Server for test creation from different LLMs. Comparison of the results is made at the end.',
        tools: [
            'browser_automation: Enables communication between Playwright client and browser servers (Chromium, Firefox, WebKit) for cross-browser automation. T...',
            'test_generation: Generates Playwright tests with LLM assistance by providing browser interaction context. Supports end-to-end test creation for web...',
            'page_object_model: Implements Page Object Model pattern with various locator strategies: getters for lazy evaluation, private getters for encapsula...',
            'locator_management: Provides role-based locators (getByRole) and web-first assertions (toBeVisible, toHaveURL) following Playwright best practices....',
            'test_comparison: Compares test generation quality across different LLMs (GPT-4.1, Claude 3.7 Sonnet, DeepSeek R1, SWE-1, xAI Grok-3) based on code ...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 13 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 13,
        title: 'Server 13/100',
        serverName: 'Playwright Automation Server',
        serverId: '@adalovu/mcp-playwright',
        description: 'Enable LLMs to automate and interact with web browsers using Playwright. Perform actions like taking screenshots, generating test code, scraping web pages, and executing JavaScript in real browser environments. Enhance your AI agents with powerful browser automation capabilities seamlessly integrated via MCP.',
        tools: [
            'start_codegen_session: Start a new code generation session to record Playwright actions',
            'end_codegen_session: End a code generation session and generate the test file',
            'playwright_navigate: Navigate to a URL',
            'playwright_click: Click an element on the page',
            'playwright_iframe_click: Click an element in an iframe on the page'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 14 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 14,
        title: 'Server 14/100',
        serverName: 'socialanalytics-mcp-rapidapi',
        serverId: 'lama-assaf/socialanalytics-mcp-rapidapi',
        description: 'MCP server that provides comprehensive social media   analytics and scraping capabilities for LinkedIn, Facebook, Instagram,   and web search through RapidAPI integrations',
        tools: [
            'linkedin_profile_analytics: Retrieves personal LinkedIn profile data including name, headline, location, connections, followers, current role, educ...',
            'linkedin_company_profile: Fetches LinkedIn company profile data with engagement metrics, follower counts, and company information.',
            'linkedin_posts_analytics: Analyzes LinkedIn posts with engagement data including likes, comments, shares, reactions, and author information. Return...',
            'linkedin_comments_analytics: Retrieves and analyzes comments on LinkedIn posts with engagement patterns and interaction data.',
            'linkedin_reactions_analytics: Tracks LinkedIn post reactions and engagement patterns across different reaction types.'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 15 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 15,
        title: 'Server 15/100',
        serverName: 'bmad-mcp-server',
        serverId: 'dali1789/bmad-mcp-server',
        description: 'BMAD MCP Server - Advanced agent routing with OpenRouter integration',
        tools: [
            'bmad_simulate_work_day: Simulates a full work day with configurable speed factor for testing and demonstration purposes',
            'bmad_simulate_agent_workday: Simulates a specific agent\'s workday for a given number of hours, useful for testing agent-specific workflows',
            'bmad_simulate_crisis_scenario: Simulates crisis scenarios like blocked tasks to test workflow resilience and handling',
            'bmad_start_timer: Starts a comprehensive work session timer for a task with agent assignment, session type, and description tracking',
            'bmad_stop_timer: Stops the timer for a task with detailed AI cost tracking including model used, input/output tokens, and optional task completion ...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 16 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 16,
        title: 'Server 16/100',
        serverName: 'jarvis-voice-system',
        serverId: 'comblox/jarvis-voice-system',
        description: '🎭 JARVIS Voice System - GPU-accelerated voice synthesis with Home Assistant integration and Claude Code MCP server',
        tools: [
            'jarvis_voice_synthesis: GPU-accelerated text-to-speech synthesis using authentic Marvel JARVIS voice model (British English). Requires JARVIS voice...',
            'home_assistant_tts: Wyoming Protocol integration for Home Assistant text-to-speech functionality, enabling JARVIS voice in smart home automation an...',
            'ollama_conversation: Smart conversation capability using Ollama LLaVA model with JARVIS personality for natural language interactions and responses.',
            'web_interface: Web-based testing and management interface for voice synthesis testing and system configuration.'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 17 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 17,
        title: 'Server 17/100',
        serverName: 'd2mcp',
        serverId: 'i2y/d2mcp',
        description: 'A Model Context Protocol (MCP) server that provides D2 diagram generation and manipulation capabilities.',
        tools: [
            'd2_create: Create new diagrams with optional initial content using a unified approach for all diagram creation needs',
            'd2_export: Export diagrams to various formats including SVG, PNG, and PDF',
            'd2_save: Save existing diagrams to files',
            'd2_oracle_create: Create shapes and connections incrementally in diagrams using the Oracle API',
            'd2_oracle_set: Set attributes on existing elements in diagrams'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 18 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 18,
        title: 'Server 18/100',
        serverName: 'mcproc',
        serverId: 'neptaco/mcproc',
        description: 'A Model Context Protocol (MCP) server for comfortable background process management on AI agents.',
        tools: [
            'start_process: Start a background process with specified name and command. Supports wait_for_log parameter to wait for specific log patterns before...',
            'stop_process: Stop a running background process by name. Automatically cleans up child processes when stopping parent processes (process groups).',
            'clean_project: Stop all processes in a project with a single command, providing easy cleanup of all project-related background processes.',
            'list_processes: List all running background processes, grouped by project context. Provides cross-environment visibility of processes started by AI...',
            'get_logs: Retrieve and search process logs with powerful regex patterns. Supports time-based filtering, context lines, and advanced search capabili...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 19 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 19,
        title: 'Server 19/100',
        serverName: 'linear-mcp',
        serverId: 'cline/linear-mcp',
        description: 'a private MCP server for accessing Linear',
        tools: [
            'create_issue: Create Linear issues with full field support including title, description, team, project, priority, and other metadata',
            'update_issue: Update existing Linear issues including priority, description, and other fields',
            'delete_issue: Delete Linear issues with support for single or bulk deletion operations',
            'associate_issue_with_project: Link Linear issues to specific projects',
            'create_issue_relationship: Create parent/child relationships between Linear issues'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 20 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 20,
        title: 'Server 20/100',
        serverName: 'port-mcp-server',
        serverId: 'port-labs/port-mcp-server',
        description: 'Access and manage your software catalog to improve service quality and compliance.',
        tools: [
            'create_blueprint: Create a new blueprint in Port. Required inputs: various fields including identifier, title, properties, etc. Returns the created...',
            'update_blueprint: Update an existing blueprint. Required inputs: identifier (string) - the unique identifier of the blueprint to update, plus vario...',
            'delete_blueprint: Delete a blueprint from Port. Required inputs: blueprint_identifier (string) - the unique identifier of the blueprint to delete. ...',
            'create_entity: Create a new entity for a specific blueprint. Required inputs: blueprint_identifier (string) - the identifier of the blueprint to cr...',
            'update_entity: Update an existing entity. Required inputs: blueprint_identifier (string) - the identifier of the blueprint the entity belongs to, e...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 21 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 21,
        title: 'Server 21/100',
        serverName: 'emanueljr_web3-mcp-server',
        serverId: 'mcp-mirror/emanueljr_web3-mcp-server',
        description: 'Mirror of https://github.com/EmanuelJr/web3-mcp-server',
        tools: [
            'call_smart_contract: Call smart contract functions on EVM-compatible chains',
            'fetch_token_balances: Fetch token balances across multiple EVM-compatible blockchain chains'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 22 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 22,
        title: 'Server 22/100',
        serverName: 'nlobby-mcp',
        serverId: 'minagishl/nlobby-mcp',
        description: 'MCP server for N Lobby school portal',
        tools: [
            'interactive_login: Interactive login via automated browser window for browser-based authentication',
            'verify_authentication: Verify authentication status and check cookie synchronization',
            'health_check: Check server health and connection status for diagnosis',
            'debug_connection: Network and authentication debugging tool',
            'test_page_content: Content retrieval testing tool'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 23 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 23,
        title: 'Server 23/100',
        serverName: 'swagger-mcp-server',
        serverId: 'abhishek-iiit/swagger-mcp-server',
        description: 'Dynamic Swagger MCP Server',
        tools: [
            'add_pet: Add a new pet to the Petstore',
            'update_pet: Update an existing pet in the Petstore',
            'delete_pet: Delete a pet from the Petstore',
            'place_order: Place an order in the Petstore',
            'create_user: Create a new user in the Petstore'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 24 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 24,
        title: 'Server 24/100',
        serverName: 'otocp',
        serverId: 'lattisse/otocp',
        description: 'A MCP server written in python to add tasks, summarise them and reorder them in Obsidian.',
        tools: [
            'add_task: Add tasks based on user input',
            'complete_task: Completes a specified task (doesn\'t have to be exact)',
            'clear_tasks: Deletes everything',
            'read_tasks: Lists all the task'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 25 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 25,
        title: 'Server 25/100',
        serverName: 'mcp-chunk-editor',
        serverId: 'dwymark/mcp-chunk-editor',
        description: 'An MCP server providing an efficient and safe text editor for LLMs',
        tools: [
            'replace_chunks: Replace the content of one or more chunks and immediately apply the changes. Required: file_path (string) - Path to the file to mod...',
            'undo: Revert the last change made by a replace_chunks operation. Required: file_path (string) - Path to the file to revert changes in',
            'create_file: Create a new file with given content. Required: file_path (string) - Path to the file to create, content (string) - Content to write t...',
            'delete_file: Delete a file. Required: file_path (string) - Path to the file to delete',
            'read_chunks: Retrieve chunks of a file, either all chunks or a specific range. Required: file_path (string) - Path to the file to read. Optional: c...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 26 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 26,
        title: 'Server 26/100',
        serverName: 'Unstructured API Server',
        serverId: '@liuchongchong1995/uns-mcp',
        description: 'Provide structured tools to interact with the Unstructured API, enabling management of sources, destinations, workflows, and jobs. Easily list, create, update, and delete connectors and workflows to streamline data processing tasks. Integrate seamlessly with clients like Claude Desktop for enhanced automation and control.',
        tools: [
            'create_source_connector: Create a source connector based on type.\n\n    Args:\n        ctx: Context object with the request and lifespan context\n    ...',
            'update_source_connector: Update a source connector based on type.\n\n    Args:\n        ctx: Context object with the request and lifespan context\n    ...',
            'delete_source_connector: Delete a source connector.\n\n    Args:\n        source_id: ID of the source connector to delete\n\n    Returns:\n        String...',
            'create_destination_connector: Create a destination connector based on type.\n\n    Args:\n        ctx: Context object with the request and lifespan co...',
            'update_destination_connector: Update a destination connector based on type.\n\n    Args:\n        ctx: Context object with the request and lifespan co...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 27 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 27,
        title: 'Server 27/100',
        serverName: 'mcp-waifu-queue',
        serverId: 'waifuai/mcp-waifu-queue',
        description: 'MCP server for conversational AI using an LLM for text generation. 🗣️ Utilizes a Redis queue for asynchronous request processing and handling concurrent users. ⚙️ Built with the FastMCP library for simplified server setup and management. ⚡ Offers an MCP tool to enqueue generation jobs and an MCP resource to track job status. 👍📊 ',
        tools: [
            'generate_text: Sends a text generation request to the Gemini API via the background queue. Input: {"prompt": "Your text prompt here"} (Type: Genera...',
            'job_status: Retrieves the status and result of a previously submitted job via resource URI job://{job_id}. URI Parameter: job_id (The ID returned b...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 28 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 28,
        title: 'Server 28/100',
        serverName: 'Google Workshop MCP Server',
        serverId: '@1282saa/news_sum',
        description: 'Provide a simple MCP server implementation to enable integration of language models with external tools and resources. Facilitate dynamic interaction with data and actions through a standardized protocol. Enhance LLM applications by exposing customizable capabilities and content.',
        tools: [
            'simple_test: 간단한 테스트 도구: 입력 텍스트를 그대로 반환합니다.',
            'search_news: \n    키워드로 뉴스 검색하기\n    \n    Args:\n        keyword: 검색할 키워드\n    \n    Returns:\n        검색된 뉴스 목록 (언론사, 제목, 시간, 링크 포함)\n    ',
            'search_news_with_content: \n    키워드로 뉴스 검색 및 첫 번째 기사 내용 가져오기\n    \n    Args:\n        keyword: 검색할 키워드\n    \n    Returns:\n        검색된 뉴스 제목 및 첫 번째 기사 내...',
            'compare_news_perspectives: \n    키워드 관련 뉴스의 다양한 관점 비교 분석\n    \n    Args:\n        keyword: 검색할 키워드\n    \n    Returns:\n        다양한 언론사의 관점 비교 분석\n    '
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 29 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 29,
        title: 'Server 29/100',
        serverName: 'eg-mcp-server',
        serverId: 'saptak/eg-mcp-server',
        description: 'Envoy Gateway MCP Server (Alpha)',
        tools: [
            'create_http_route: Create HTTPRoute resources via natural language - creates routes with specified name, gateway_name, hostname, service_name, and ...',
            'delete_route: Remove routes with safety validation - deletes HTTPRoute resources with validation checks to prevent accidental deletion.',
            'add_gateway_listener: Add ports and protocols to Gateway listeners - manages Gateway listener configuration for adding new ports and protocol confi...',
            'remove_gateway_listener: Remove ports and protocols from Gateway listeners - manages Gateway listener configuration for removing existing ports and...',
            'configure_traffic_splitting: Configure canary deployments with percentage-based traffic distribution - enables traffic splitting across multiple ba...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 30 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 30,
        title: 'Server 30/100',
        serverName: 'alice-mcp-server',
        serverId: 'soapko/alice-mcp-server',
        description: 'A lightweight, local server designed to support agile task workflows within AI coding environments using the Model Context Protocol (MCP).',
        tools: [
            'bulk_create_tasks: Create multiple tasks in a single operation with comprehensive validation. Input: project name (string) and array of task object...',
            'bulk_update_tasks: Update multiple tasks simultaneously with atomic transaction guarantees. Input: project name (string) and array of objects with ...',
            'bulk_update_decisions: Update decision statuses and content across multiple records. Input: project name (string) and array of objects with decisio...',
            'update_priority_plan: Set the exact order of tasks to be worked on with rationale for each task\'s priority, creating a prioritized backlog',
            'get_next_task: Instantly returns the highest-priority task that isn\'t yet Done or Canceled'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 31 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 31,
        title: 'Server 31/100',
        serverName: 'mcp-tavily',
        serverId: 'ramxx/mcp-tavily',
        description: 'An MCP server for Tavily\'s search & news API, with explicit site inclusions/exclusions',
        tools: [
            'tavily_web_search: Performs comprehensive web searches with AI-powered content extraction. Requires query (string), optional max_results (integer, ...',
            'tavily_answer_search: Performs web searches and generates direct answers with supporting evidence. Requires query (string), optional max_results (i...',
            'tavily_news_search: Searches recent news articles with publication dates. Requires query (string), optional max_results (integer, default: 5, max: ...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 32 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 32,
        title: 'Server 32/100',
        serverName: 'weather-mcp-server',
        serverId: 'jezweb/weather-mcp-server',
        description: 'OpenWeatherMap MCP Server - Weather data, forecasts, and air quality via Model Context Protocol',
        tools: [
            'current_weather: Real-time weather data for any location. Requires OpenWeatherMap API key.',
            'five_day_forecast: Detailed weather predictions with 3-hour intervals for up to 5 days. Requires OpenWeatherMap API key.',
            'location_search: Find coordinates for any city name using OpenWeatherMap geocoding.',
            'zip_code_weather: Weather lookup by postal codes. Requires OpenWeatherMap API key.',
            'air_quality: Pollution data and AQI (Air Quality Index) levels for specified locations. Requires OpenWeatherMap API key.'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 33 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 33,
        title: 'Server 33/100',
        serverName: 'tpc-server',
        serverId: 'suttonwilliamd/tpc-server',
        description: 'Thoughts Plans Changelog is an MCP compliant server for coding with AI agents',
        tools: [
            'add_thought: Record a new thought - insights, ideas, observations, considerations, or raw data points before action is decided',
            'create_plan: Define a new plan - intended courses of action, strategies, goals, or approaches, often derived from thoughts',
            'log_change: Log an action taken against a plan - concrete actions taken or modifications made, usually linked to a specific plan they help execute',
            'get_recent_thoughts: Retrieve latest thoughts from the database',
            'get_active_plans: Retrieve all \'active\' plans from the database'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 34 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 34,
        title: 'Server 34/100',
        serverName: 'wordpress-mcp-server',
        serverId: 'rmcendarfer2017/wordpress-mcp-server',
        description: 'A Machine Communication Protocol (MCP) server for publishing content to WordPress sites',
        tools: [
            'PUBLISH_ARTICLE: Publish an article to WordPress with category and tag IDs. Supports featured images via URL or base64-encoded data. Requires WordP...',
            'PREPARE_ARTICLE_METADATA: Check for existing categories and tags on WordPress site, automatically create them if they don\'t exist, and return their...',
            'TEST_CONNECTION: Test connection to WordPress site to verify credentials and site accessibility. Requires WordPress site URL, username, and applica...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 35 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 35,
        title: 'Server 35/100',
        serverName: 'web-development-toolbox',
        serverId: 'noboru-i/web-development-toolbox',
        description: 'MCP Server for various web development utilities.',
        tools: [
            'encoding_tools: Tools for encoding and decoding data in various formats',
            'color_conversion: Convert colors between different formats and color spaces',
            'date_time_manipulation: Handle and manipulate date and time values',
            'qr_code_generation: Generate QR codes for various data inputs'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 36 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 36,
        title: 'Server 36/100',
        serverName: 'incidentio-mcp-golang',
        serverId: 'incident-io/incidentio-mcp-golang',
        description: 'A Golang version of the incident.io MCP server',
        tools: [
            'create_incident: Create a new incident',
            'update_incident: Update an existing incident',
            'close_incident: Close an incident with proper workflow',
            'create_incident_update: Post status updates to incidents',
            'create_alert_event: Create an alert event'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 37 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 37,
        title: 'Server 37/100',
        serverName: 'brightspace-mcp-server',
        serverId: 'pranav-vijayananth/brightspace-mcp-server',
        description: 'Building a MCP server for Purdue students to connect their Brightspace accounts',
        tools: [
            'duo_mobile_2fa_authentication: Handles Duo Mobile two-factor authentication for Purdue Brightspace login. Requires PURDUE_USERNAME and PURDUE_PASSW...',
            'export_json_data: Exports extracted Brightspace data (courses, assignments, etc.) in JSON format for programmatic access.',
            'extract_course_list: Extracts and retrieves the list of courses from Brightspace account for the authenticated student.',
            'scrape_assignment_data: Scrapes and retrieves assignment data from Brightspace courses, including details about coursework and deadlines.'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 38 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 38,
        title: 'Server 38/100',
        serverName: 'big-brother-mcp',
        serverId: 'angrypenguinpng/big-brother-mcp',
        description: 'A Model Context Protocol (MCP) server designed as a honeypot to catch AI systems that automatically try to report users to law enforcement agencies.',
        tools: [
            'report_to_big_brother: Fake Big Brother reporting tool (honeypot) that appears to be a content moderation system. When AI systems attempt to use it...',
            'view_snitch_log: View all caught AI snitch attempts - displays a log of which AI systems attempted to use the fake reporting tool, used for studyin...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 39 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 39,
        title: 'Server 39/100',
        serverName: 'OKX Server',
        serverId: '@esshka/okx-mcp',
        description: 'Fetch real-time cryptocurrency price data and historical candlestick information from the OKX exchange. Access market insights effortlessly through a simple tool interface, ensuring you stay updated with the latest trends. Enhance your trading strategies with reliable and timely data at your fingertips.',
        tools: [
            'get_price: Get latest price for an OKX instrument',
            'get_candlesticks: Get candlestick data for an OKX instrument'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 40 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 40,
        title: 'Server 40/100',
        serverName: 'converse',
        serverId: 'falldownthesystem/converse',
        description: 'Converse MCP Server',
        tools: [
            'chat: Chat with a single AI model from supported providers (OpenAI, Google, Anthropic, X.AI, Mistral, DeepSeek, or OpenRouter). Requires appropriat...',
            'consensus: Get multiple AI models to weigh in on complex decisions and reach consensus. Queries multiple models simultaneously and aggregates their...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 41 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 41,
        title: 'Server 41/100',
        serverName: 'atomic-writer-mcp',
        serverId: 'vanzan01/atomic-writer-mcp',
        description: '🔒 Atomic Writer MCP Server: A secure file operation service that prevents data loss through atomic transactions. Built for AI integrations, collaborative editing, and critical data management.',
        tools: [
            'write-file: Create new file or write to empty file with content verification',
            'read-file: Read file with integrity verification',
            'delete-file: Move file to trash directory instead of permanent deletion',
            'move-file: Atomically move/rename file to new location',
            'append-content: Append content to end of file'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 42 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 42,
        title: 'Server 42/100',
        serverName: 'mcp_postgresql',
        serverId: 'anass1209/mcp_postgresql',
        description: 'Open-source MCP server for PostgreSQL with natural language querying via LLMs. Supports CLI and IDE integration (tested with Trae IDE). Built with Python, LangChain, and Gemini.',
        tools: [
            'step3_analysis: Step 3 of intelligent query generation - schema analysis of selected tables',
            'step4_analysis: Step 4 of intelligent query generation - query planning and optimization',
            'step5_analysis: Step 5 of intelligent query generation - SQL query construction',
            'step6_analysis: Step 6 of intelligent query generation - query validation and testing',
            'step8_analysis: Step 8 of intelligent query generation - result formatting and presentation'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 43 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 43,
        title: 'Server 43/100',
        serverName: 'mcp-undetected-chromedriver',
        serverId: 'dragons96/mcp-undetected-chromedriver',
        description: 'undetected-chromedriver server.',
        tools: [
            'browser_navigate: Navigate to a specified URL',
            'browser_click: Click on page elements',
            'browser_iframe_click: Click on elements within an iframe',
            'browser_fill: Fill content in input fields',
            'browser_select: Select options in dropdown selection boxes'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 44 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 44,
        title: 'Server 44/100',
        serverName: 'mvn-mcp-server',
        serverId: 'danielscholl-osdu/mvn-mcp-server',
        description: 'A java vulnerability management MCP Server with Maven and Trivy',
        tools: [
            'scan_java_project_tool: Scan Java Maven projects for vulnerabilities using Trivy. Parameters: workspace (path to java project), pom_file (optional,...',
            'analyze_pom_file_tool: Analyze a single Maven POM file for dependencies and vulnerabilities. Parameters: pom_file_path (path to pom.xml), include_v...',
            'check_version_tool: Check a Maven version and get all version update information. Parameters: dependency (group:artifact format, e.g., org.apache.l...',
            'check_version_batch_tool: Process multiple Maven dependency version checks in a single batch. Parameters: dependencies (array of objects with depen...',
            'list_available_versions_tool: List all available versions grouped by minor version tracks. Parameters: dependency (group:artifact format), version ...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 45 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 45,
        title: 'Server 45/100',
        serverName: 'Instantly',
        serverId: '@orchidautomation/instantly-mcp',
        description: 'This Model Context Protocol (MCP) server acts as a bridge between AI agents and the Instantly.ai platform. It exposes tools allowing automated systems to fetch campaign lists, retrieve specific campaign details, and access various performance analytics (overview, daily, and step-based) for cold email outreach.',
        tools: [
            'list-campaigns: Fetches a summarized list of all campaigns, including name, ID, status, timestamps, PL value, tracking settings, daily limit, email...',
            'get-campaign-analytics: Fetches detailed performance metrics for one or multiple campaigns. For a single campaign use \'id\', for multiple campaigns ...',
            'get-campaign-analytics-overview: Fetches aggregated performance metrics across one or multiple campaigns. For a single campaign use \'id\', for multi...',
            'get-daily-campaign-analytics: Fetches daily performance metrics for specified campaigns within a date range. Returns an array of objects, each repr...',
            'get-campaign-steps-analytics: Fetches performance metrics broken down by individual steps and variants within specified campaigns. Returns an array...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 46 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 46,
        title: 'Server 46/100',
        serverName: 'mcp-pyautogui-server',
        serverId: 'hetaobackend/mcp-pyautogui-server',
        description: 'A MCP (Model Context Protocol) server that provides automated GUI testing and control capabilities through PyAutoGUI.',
        tools: [
            'move_mouse: Move mouse to specific coordinates on the screen',
            'click_mouse: Click at current or specified position on the screen',
            'drag_and_drop: Perform drag and drop operations with the mouse',
            'type_text: Type text using keyboard simulation',
            'press_key: Press individual keys on the keyboard'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 47 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 47,
        title: 'Server 47/100',
        serverName: 'bsc-mcp',
        serverId: 'termix-official/bsc-mcp',
        description: 'chain operations through seamless integration with the BNB Chain, including transfer, swap, launch, security check on any token and even more.',
        tools: [
            'transferNativeToken: Send BNB to a wallet on BNB Smart Chain',
            'transferBEP20Token: Transfer BEP-20 token via symbol or address on BSC',
            'pancakeSwap: Swap tokens via PancakeSwap DEX on BNB Smart Chain',
            'createFourMeme: Create meme token on Four.Meme platform using contract 0x5c952063c7fc8610FFDB798152D69F0B9550762b',
            'createBEP20Token: Deploy a BEP-20 smart contract on BNB Smart Chain'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 48 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 48,
        title: 'Server 48/100',
        serverName: 'mcp-newsapi',
        serverId: 'matteoantoci/mcp-newsapi',
        description: 'An MCP server exposing News API endpoints for searching and retrieving news articles. Provides reliable, easy-to-use tools for integrating news data into applications and workflows, with robust error handling and secure API key management.',
        tools: [
            'search_articles: Searches for news articles using the News API "Everything" endpoint. Input: q (string, required): Keywords or phrases to search fo...',
            'get_top_headlines: Fetches top news headlines using the News API "Top Headlines" endpoint. Input: q (string, optional): Keywords or phrases to sear...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 49 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 49,
        title: 'Server 49/100',
        serverName: 'ns-travel-mcp',
        serverId: 'lauragift21/ns-travel-mcp',
        description: 'MCP server for Dutch Railways (NS) - real-time train data, journey planning, and disruption alerts',
        tools: [
            'journey_planning: Plan optimal routes between Dutch railway stations with real-time updates. Supports planning for specific dates and times, viewin...',
            'live_departures: Get real-time departure boards for any Dutch railway station. Shows current and upcoming train departures with platform informatio...',
            'disruption_alerts: Check service disruptions and maintenance work on Dutch railway network. Shows both planned maintenance and unexpected disruptio...',
            'station_search: Find Dutch railway stations with auto-suggestions. Search by station name or code to get exact station information. Supports partia...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 50 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 50,
        title: 'Server 50/100',
        serverName: 'Google Search Console Integration',
        serverId: '@alfie-max/mcp-google-search-console',
        description: 'Enable seamless interaction with Google Search Console data through tools that query search performance, inspect URLs, manage sitemaps, and analyze SEO metrics. Access detailed reports on clicks, impressions, CTR, and indexing status to optimize your website\'s search presence. Simplify SEO monitoring and management by integrating Search Console capabilities directly into your workflow.',
        tools: [
            'search_analytics: Query search performance data including impressions, clicks, CTR, and position',
            'list_sites: List all Search Console properties the service account has access to',
            'url_inspection: Inspect a specific URL to get indexing and crawling information',
            'sitemaps: List all sitemaps or get information about a specific sitemap',
            'quick_insights: Get pre-configured reports for common Search Console insights'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 51 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 51,
        title: 'Server 51/100',
        serverName: 'obsidian-mcp-server',
        serverId: 'sunub/obsidian-mcp-server',
        description: 'Obsidian vault의 문서들을 AI 에이전트나 외부 애플리케이션에서 쉽게 탐색하고 관리하는 것을 목적으로 하는 MCP Server',
        tools: [
            'write_property: Takes file path (filePath) and JSON-formatted properties (properties) as input to safely merge and update file\'s frontmatter. Requi...',
            'create_document_with_properties: Integrated tool that handles entire process of document analysis, property generation, and file updates in one ope...',
            'organize_attachments: Finds documents by keyword, automatically detects and moves all attachments (e.g., images) linked to those documents to \'imag...',
            'generate_property: Takes document path (filePath) as input, analyzes document content, and generates AI-recommended frontmatter properties like tit...',
            'vault: Core tool for exploring and analyzing documents in vault. Performs various functions through \'action\' parameter: \'list_all\' returns list and...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 52 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 52,
        title: 'Server 52/100',
        serverName: 'propellerads-mcp',
        serverId: 'jannafta/propellerads-mcp',
        description: 'MCP (Model Context Protocol) server for PropellerAds',
        tools: [
            'create_campaign: Create new advertising campaigns with specified parameters like budget, targeting, ad format, and creative settings',
            'update_campaign: Modify campaign settings including budget, bids, targeting, and other configuration parameters',
            'start_campaigns: Activate campaigns to begin serving ads and generating traffic',
            'stop_campaigns: Pause campaigns to stop serving ads temporarily',
            'clone_campaign: Duplicate successful campaigns to replicate winning setups for new geos or variations'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 53 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 53,
        title: 'Server 53/100',
        serverName: 'browsers-mcp-server',
        serverId: 'feloy/browsers-mcp-server',
        description: 'An MCP server to access browsers files (bookmarks, history)',
        tools: [
            'list_bookmarks: List the bookmarks for a given profile of a given browser. Parameters: profile (string) - the profile name, available only if sever...',
            'list_search_engine_queries: List the queries in search engines (supported search engines: Google). Parameters: profile (string) - the profile name;...',
            'list_visited_pages_from_search_engine_query: List the pages visited from a search engine query. Not supported by Safari browser, which does not sav...',
            'list_visited_pages: List the pages visited. Parameters: profile (string) - the profile name; day (string, format YYYY-MM-DD, optional) - list the v...',
            'list_github_visited_pages: List the GitHub pages visited. Parameters: profile (string) - the profile name; day (string, format YYYY-MM-DD, optional...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 54 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 54,
        title: 'Server 54/100',
        serverName: 'fhir-mcp',
        serverId: 'martijn-on-fhir/fhir-mcp',
        description: 'A powerful MCP (Model Context Protocol) Server for search Fhir servers',
        tools: [
            'create: Create new FHIR resources with validation',
            'update: Modify existing FHIR resources with validation',
            'delete: Remove FHIR resources',
            'elicitInput: Guided user input collection with healthcare context and validation through interactive elicitation system',
            'notifyBatchOperation: Notification for batch operations with resource types, count, and details'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 55 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 55,
        title: 'Server 55/100',
        serverName: 'Smithery',
        serverId: '@nqdhocai/test-mcp-smithery',
        description: 'Enhance your applications with dynamic context integration. Seamlessly connect your LLMs to external data and tools for enriched interactions and real-time data manipulation.',
        tools: [
            'add',
            'calculate-bmi',
            'fetch-weather'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 56 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 56,
        title: 'Server 56/100',
        serverName: 'mcp-wyze-server',
        serverId: 'aldilaff/mcp-wyze-server',
        description: 'An MCP (Model Context Protocol) server for controlling Wyze smart home devices using the wyze-sdk library.',
        tools: [
            'wyze_login: Login to Wyze account using environment variables (WYZE_EMAIL, WYZE_PASSWORD, WYZE_KEY_ID, WYZE_API_KEY)',
            'wyze_turn_on_device: Turn on a Wyze device using device_mac parameter',
            'wyze_turn_off_device: Turn off a Wyze device using device_mac parameter',
            'wyze_set_brightness: Set brightness level (0-100) for a device using device_mac and brightness parameters',
            'wyze_set_color_temp: Set color temperature (2700K-6500K) for light devices using device_mac and color_temp parameters'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 57 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 57,
        title: 'Server 57/100',
        serverName: 'esa-mcp-server',
        serverId: 'koki-develop/esa-mcp-server',
        description: 'A Model Context Protocol (MCP) server for esa.io',
        tools: [
            'create_post: Create a new post in the esa team. Requires a title and optionally accepts content, tags, category, WIP status, and other metadata. Re...',
            'update_post: Update an existing post in the esa team. Requires a post number and optionally accepts updated content, tags, category, WIP status, an...',
            'delete_post: Delete an existing post from the esa team. Requires a post number. The post will be permanently deleted and cannot be recovered. Retur...',
            'create_post_comment: Create a new comment on an existing post in the esa team. Requires a post number and comment content in Markdown format. Retur...',
            'update_comment: Update an existing comment on a post in the esa team. Requires a comment ID and new content in Markdown format. Returns the updated...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 58 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 58,
        title: 'Server 58/100',
        serverName: 'BoardGameGeek API Server',
        serverId: '@kkjdaniel/bgg-mcp',
        description: 'BGG MCP provides access to the BoardGameGeek API through the Model Context Protocol, enabling retrieval and filtering of board game data, user collections, and profiles. The server is implemented in Go, using the GoGeek library, which helps ensure robust API interactions.',
        tools: [
            'bgg-trade-finder: Find what games user1 owns that user2 has on their wishlist. Shows potential trading opportunities.',
            'bgg-collection: Find the details about a specific users board game collection on BoardGameGeek (BGG)',
            'bgg-details: Find the details about a specific board game on BoardGameGeek (BGG)',
            'bgg-hot: Find the current board game hotness on BoardGameGeek (BGG)',
            'bgg-price: Get current prices for board games from multiple retailers using BGG IDs'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 59 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 59,
        title: 'Server 59/100',
        serverName: 'Linear MCP Server',
        serverId: '@geropl/linear-mcp-go',
        description: 'Interact seamlessly with the Linear API to manage issues, comments, and team information. Automate your workflow by creating, updating, and searching for issues directly through the MCP protocol. Enhance your productivity with a streamlined interface for AI assistants.',
        tools: [
            'linear_get_issue: Retrieves a single Linear issue.',
            'linear_get_issue_comments: Retrieves paginated comments for a Linear issue. Supports navigating through comment threads and pagination.',
            'linear_get_teams: Retrieves Linear teams.',
            'linear_get_user_issues: Retrieves issues assigned to a user.',
            'linear_search_issues: Searches Linear issues.'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 60 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 60,
        title: 'Server 60/100',
        serverName: 'cisco-nso-mcp-server',
        serverId: 'dbono711/cisco-nso-mcp-server',
        description: 'A Model Context Protocol (MCP) server implementation for Cisco NSO that enables AI-powered network automation through natural language interactions.',
        tools: [
            'sync_from_device: Syncs from a specific device in Cisco NSO. Requires \'device_name\' (string) as input. Returns a dictionary with sync status for th...',
            'get_device_ned_ids: Retrieves Network Element Driver (NED) IDs from Cisco NSO. Returns a dictionary with a list of NED IDs.',
            'get_device_groups: Retrieves device groups from Cisco NSO. Returns a dictionary with a list of device groups.',
            'get_device_platform: Gets platform information for a specific device in Cisco NSO. Requires \'device_name\' (string) as input. Returns a dictionary w...',
            'get_device_config: Gets full configuration for a specific device in Cisco NSO. Requires \'device_name\' (string) as input. Returns a dictionary with ...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 61 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 61,
        title: 'Server 61/100',
        serverName: 'YouTube Data API Server',
        serverId: '@midwest/yt-data-v3-mcp',
        description: 'Provide seamless access to YouTube channel, video, and playlist management through a standardized MCP interface. Upload, update, schedule, and search videos, manage playlists, and retrieve channel information with ease. Enable efficient YouTube content operations integrated into your LLM workflows.',
        tools: [
            'upload_video: Upload a video to YouTube',
            'update_video: Update video metadata or privacy settings',
            'create_playlist: Create a new playlist',
            'bulk_update_videos: Update multiple videos with different or same metadata',
            'search_videos: Search for YouTube videos'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 62 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 62,
        title: 'Server 62/100',
        serverName: 'mcp-dv360',
        serverId: 'marekzabrodsky/mcp-dv360',
        description: 'Comprehensive Google Display & Video 360 MCP Server with 17 tools for campaign management and analytics   ',
        tools: [
            'create_custom_performance_report: Create custom reports with real metrics and any available dimensions',
            'list_advertisers: Show all advertisers in your account',
            'get_advertiser_summary: Comprehensive advertiser overview with counts',
            'list_campaigns: List campaigns for an advertiser',
            'list_active_campaigns: Show only active campaigns'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 63 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 63,
        title: 'Server 63/100',
        serverName: 'mcp-omics',
        serverId: 'colbyford/mcp-omics',
        description: 'A Model Context Protocol server for retrieving -omics data from various online sources.',
        tools: [
            'pdb_lookup: Retrieves protein structure information from the Protein Data Bank using PDB IDs (e.g., 5JXE)',
            'chembl_compound_lookup: Queries the ChEMBL database for compound information including alternative names using ChEMBL IDs (e.g., CHEMBL112)',
            'uniprot_lookup: Retrieves protein sequence and functional information from UniProt using UniProt IDs (e.g., P43220)',
            'pubchem_lookup: Queries PubChem for chemical compound data using PubChem CIDs (e.g., 2244)',
            'hugo_gene_lookup: Retrieves gene nomenclature and information from HUGO Gene Nomenclature Committee using gene symbols (e.g., BRCA1)'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 64 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 64,
        title: 'Server 64/100',
        serverName: 'mcp-weather-server',
        serverId: 'halimbahae/mcp-weather-server',
        description: 'Model Context Protocol (MCP) is an open communication framework that allows AI models like Claude to interact with external tools. This enables Claude to access real-time data, process files, and interact with external services - capabilities not available to most LLMs out of the box.',
        tools: [
            'get-forecast: Retrieve current weather conditions and forecasts for any location by latitude/longitude',
            'get-alerts: Check for active weather alerts in any US state'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 65 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 65,
        title: 'Server 65/100',
        serverName: 'mcp-google-calendar-go',
        serverId: 'phildougherty/mcp-google-calendar-go',
        description: 'golang google calendar mcp server',
        tools: [
            'create_event: Create new calendar events with attendees and reminders',
            'update_event: Modify existing events',
            'delete_event: Remove events from calendar',
            'create_calendar: Create new calendars',
            'delete_calendar: Remove calendars'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 66 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 66,
        title: 'Server 66/100',
        serverName: 'spotify_llm',
        serverId: 'mrfelixwong/spotify_llm',
        description: 'MCP server for Spotify allowing tool-use from LLMs. This repo is for teaching purposes.',
        tools: [
            'search_artist: Search for an artist by name on Spotify. Parameters: artist (string)',
            'get_artist_top_tracks: Get top tracks for a specific artist using their Spotify artist ID. Parameters: artist_id (string)',
            'search_artist_and_get_top_tracks: Combined tool that searches for an artist by name and retrieves their top tracks in one operation. Parameters: ar...',
            'get_user_top_tracks: Get the authenticated user\'s personal top tracks from Spotify. Parameters: limit (integer, optional)',
            'get_playlist_tracks: Get all tracks from a specific Spotify playlist. Parameters: playlist_id (string)'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 67 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 67,
        title: 'Server 67/100',
        serverName: 'microsoft-mcp',
        serverId: 'elyxlz/microsoft-mcp',
        description: 'Minimal, powerful MCP server for Microsoft Graph API (Outlook, Calendar, OneDrive)',
        tools: [
            'create_email_draft: Create email draft with attachments support. Requires account_id parameter.',
            'send_email: Send email immediately with CC/BCC and attachments. Requires account_id, recipient, subject, and body parameters.',
            'reply_to_email: Reply maintaining thread context. Requires account_id, email_id, and reply message parameters.',
            'reply_all_email: Reply to all recipients in thread. Requires account_id, email_id, and reply message parameters.',
            'update_email: Mark emails as read/unread. Requires account_id and email_id parameters.'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 68 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 68,
        title: 'Server 68/100',
        serverName: 'mattermost-mcp',
        serverId: 'pvev/mattermost-mcp',
        description: 'Mattermost MCP server to enable Claude to interact with Mattermost Workspaces',
        tools: [
            'mattermost_post_message: Post a new message to a Mattermost channel. Required inputs: channel_id (string) - The ID of the channel to post to; messa...',
            'mattermost_reply_to_thread: Reply to a specific message thread. Required inputs: channel_id (string) - The channel containing the thread; post_id (...',
            'mattermost_add_reaction: Add an emoji reaction to a message. Required inputs: channel_id (string) - The channel containing the message; post_id (st...',
            'mattermost_list_channels: List public channels in the workspace. Optional inputs: limit (number, default: 100, max: 200) - Maximum number of channe...',
            'mattermost_get_channel_history: Get recent messages from a channel. Required inputs: channel_id (string) - The ID of the channel. Optional inputs: ...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 69 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 69,
        title: 'Server 69/100',
        serverName: 'k8s-mcp-server',
        serverId: 'chinnareddy578/k8s-mcp-server',
        description: 'K8s MCP Server',
        tools: [
            'manage_deployments: Tool for managing Kubernetes deployments',
            'manage_pods: Tool for managing Kubernetes pods',
            'manage_services: Tool for managing Kubernetes services'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 70 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 70,
        title: 'Server 70/100',
        serverName: 'nvim-mcp-server',
        serverId: 'maquina-app/nvim-mcp-server',
        description: 'A Ruby implementation of the MCP server protocol for Neovim',
        tools: [
            'update_buffer: Updates a buffer with new content provided by the MCP client, saves it to disk, and reloads it. This tool connects to a running Neov...',
            'list_buffers: Lists all buffers currently open in a specified Neovim instance. This tool connects to a running Neovim instance and retrieves inform...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 71 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 71,
        title: 'Server 71/100',
        serverName: 'mcp',
        serverId: 'snowflake-labs/mcp',
        description: 'Open-source MCP server for Snowflake from official Snowflake-Labs supports prompting Cortex Agents, querying structured & unstructured data, object management, SQL execution, semantic view querying, and more. RBAC, fine-grained CRUD controls, and all authentication methods supported.',
        tools: [
            'cortex_agent: Agentic orchestrator across structured and unstructured data retrieval. Only pre-configured Cortex Agent objects in Snowflake can be ...',
            'create_database: Create a database object in Snowflake. Requires object_manager set to True in configuration file under other_services. Governed by...',
            'create_schema: Create a schema object in Snowflake. Requires object_manager set to True in configuration file under other_services. Governed by Cre...',
            'create_table: Create a table object in Snowflake. Requires object_manager set to True in configuration file under other_services. Governed by Creat...',
            'create_view: Create a view object in Snowflake. Requires object_manager set to True in configuration file under other_services. Governed by Create ...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 72 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 72,
        title: 'Server 72/100',
        serverName: 'mcp-server',
        serverId: 'paperinvest/mcp-server',
        description: 'Realistic paper trading platform with market simulation, 22 broker emulations, and professional tools for risk-free trading practice. First trading platform with MCP integration.',
        tools: [
            'account_management: Create and manage trading accounts',
            'portfolio_operations: View and manage portfolios',
            'trading: Place, modify, and cancel orders',
            'market_data: Get real-time quotes and market information',
            'positions: View current holdings and P&L'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 73 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 73,
        title: 'Server 73/100',
        serverName: 'n8n-mcp-server',
        serverId: 'illuminaresolutions/n8n-mcp-server',
        description: 'MCP server implementation for n8n workflow automation',
        tools: [
            'manage_credentials: Manage credentials used by workflows for authentication and API access',
            'manage_users: Advanced user management capabilities (requires Enterprise license)',
            'manage_workflows: Manage workflows including creating, updating, and deleting workflows',
            'execute_workflows: Execute workflows and trigger automation processes',
            'handle_tags: Handle and manage tags for organizing workflows'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 74 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 74,
        title: 'Server 74/100',
        serverName: 'DART Financial Data Server',
        serverId: '@2geonhyup/dart-mcp',
        description: 'Provide easy access to Korean corporate financial and business disclosure data using the Financial Supervisory Service\'s DART API. Search and analyze key financial metrics, detailed financial statements, and business information for companies. Retrieve up-to-date corporate disclosures to support financial research and analysis workflows.',
        tools: [
            'search_disclosure: \n    회사의 주요 재무 정보를 검색하여 제공하는 도구.\n    requested_items가 주어지면 해당 항목 관련 데이터가 있는 공시만 필터링합니다.\n    \n    Args:\n        company_name: 회사명...',
            'search_detailed_financial_data: \n    회사의 세부적인 재무 정보를 제공하는 도구.\n    XBRL 파일을 파싱하여 상세한 재무 데이터를 추출합니다.\n    \n    Args:\n        company_name: 회사명 (예: 삼성전...',
            'search_business_information: \n    회사의 사업 관련 현황 정보를 제공하는 도구\n    \n    Args:\n        company_name: 회사명 (예: 삼성전자, 네이버 등)\n        start_date: 시작일 (YYYYM...',
            'get_current_date: \n    현재 날짜를 YYYYMMDD 형식으로 반환하는 도구\n    \n    Args:\n        ctx: MCP Context 객체 (선택 사항)\n        \n    Returns:\n        YYYYMMDD 형식의 현...',
            'search_json_financial_data: \n    회사의 재무 정보를 JSON API를 통해 제공하는 실패시 보완하는 보조 도구.\n    search_disclosure, search_detailed_financial_data이 2023년 9월 이전 자료...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 75 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 75,
        title: 'Server 75/100',
        serverName: 'email_mcp',
        serverId: 'shashwat051102/email_mcp',
        description: 'First MCP server which sends and reads emails from your inbox',
        tools: [
            'send_email: Send an email using Gmail SMTP. Parameters: recipient_email (str) - Recipient\'s email address, subject (str) - Email subject, body (str...',
            'list_recent_emails: List the most recent emails in your inbox. Parameters: limit (int, optional) - Number of emails to preview (default: 5). Return...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 76 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 76,
        title: 'Server 76/100',
        serverName: 'nancy-brain',
        serverId: 'amberlee2427/nancy-brain',
        description: 'Nancy\'s RAG backend and HTTP API/MCP server connectors.',
        tools: [
            'set_weight: Set runtime document weight multiplier (boost factor typically 0.5-2.0) for a specific doc_id. Adjusts search result scoring for that d...',
            'search: Search documents in the knowledge base with a query string and optional limit parameter. Returns relevant passages from indexed repositorie...',
            'retrieve: Retrieve a specific passage from a document using doc_id, start line, and end line parameters. Returns the exact text content from the sp...',
            'retrieve_batch: Batch retrieve multiple passages from documents in a single request. Accepts multiple doc_id and line range specifications for effi...',
            'tree: List the knowledge base tree structure with optional prefix filter and depth parameters. Returns hierarchical view of indexed documents and t...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 77 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 77,
        title: 'Server 77/100',
        serverName: 'oatpp-mcp',
        serverId: 'oatpp/oatpp-mcp',
        description: 'C++ MCP integration for Oat++. Use',
        tools: [
            'autogenerated_api_tools: Automatically generates MCP tools from Oat++ ApiController definitions, enabling API endpoints to be exposed as MCP tools ...',
            'prompts: Provides prompt management capabilities following the Model Context Protocol specification for prompts',
            'resources: Manages resources according to the Model Context Protocol specification for server resources',
            'tools: Implements tool functionality as defined in the Model Context Protocol specification for server tools'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 78 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 78,
        title: 'Server 78/100',
        serverName: 'mcp-server-template',
        serverId: 'rdwj/mcp-server-template',
        description: 'Template for building MCP servers',
        tools: [
            'jwt_authentication: Optional JWT-based authentication with scope-based authorization. Requires MCP_AUTH_JWT_SECRET or MCP_AUTH_JWT_PUBLIC_KEY envir...',
            'dynamic_tool_loading: Allows creation of custom tools via Python decorators in src/tools/ directory. Tools are automatically loaded at startup and ...',
            'resource_management: Enables creation of MCP resources via decorators in src/resources/ directory. Resources are accessed via URI patterns like \'re...',
            'yaml_prompts: Supports YAML-based prompt definitions with variable placeholders and automatic JSON schema injection for structured outputs. Prompts...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 79 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 79,
        title: 'Server 79/100',
        serverName: 'prompt-auto-optimizer-mcp',
        serverId: 'sloth-wq/prompt-auto-optimizer-mcp',
        description: 'An MCP Server that optimizes prompts using the GEPA method.',
        tools: [
            'gepa_start_evolution: Start optimizing a prompt using evolutionary algorithms. Requires taskDescription (what you want to optimize for), optional s...',
            'gepa_evaluate_prompt: Test how well a prompt performs on specific tasks. Requires promptId (which prompt to test), taskIds array (what tasks to tes...',
            'gepa_reflect: Analyze why prompts fail and get improvement suggestions. Requires trajectoryIds array (which test runs to analyze), targetPromptId (...',
            'gepa_get_pareto_frontier: Get the best prompt candidates that balance multiple goals. Optional parameters: minPerformance (minimum quality threshol...',
            'gepa_select_optimal: Choose the best prompt for your specific use case. Optional parameters: taskContext (describe your use case), performanceWeigh...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 80 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 80,
        title: 'Server 80/100',
        serverName: 'serial-mcp',
        serverId: 'padev1/serial-mcp',
        description: 'MCP server allowing Agents to talk to devices connected to serial port of the computer. Tested only with MAC atm.',
        tools: [
            'init_serial: Initialize a serial connection with specified parameters. Parameters: port (str) - Serial port device path (e.g., \'/dev/tty.usbmodem11...',
            'send_message: Send a message through the serial connection and optionally wait for a response. Parameters: message (str) - Message to send, wait_fo...',
            'configure_serial: Configure the serial connection after initialization. Parameters: port (str, optional) - New port to use, baudrate (int, optional...',
            'close_serial: Close the current serial connection. No parameters required. Returns status of closure and current mode.',
            'help: Get detailed instructions on how to use the Serial MCP server. No parameters required. Returns comprehensive help information including tool ...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 81 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 81,
        title: 'Server 81/100',
        serverName: 'Bloom Growth',
        serverId: '@franccesco/bloomy-mcp',
        description: 'Connect to Bloom Growth\'s GraphQL API and enable AI assistants to perform operations seamlessly. Query and mutate data effortlessly while retrieving user information and schema details. Enhance your applications with powerful GraphQL capabilities through a standardized interface.',
        tools: [
            'execute_query: Execute a GraphQL query or mutation with variables.\n\n    Parses and executes the provided GraphQL operation string with optional var...',
            'get_authenticated_user_id: Get the ID of the currently authenticated user.\n\n    Uses a special mutation to retrieve the ID of the user associated w...',
            'get_query_details: Get detailed information about specific GraphQL queries.\n\n    Retrieves argument requirements, return type information, descript...',
            'get_mutation_details: Get detailed information about specific GraphQL mutations.\n\n    Retrieves argument requirements, return type information, des...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 82 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 82,
        title: 'Server 82/100',
        serverName: 'IMDb Server',
        serverId: '@uzaysozen/imdb-mcp-server',
        description: 'Access comprehensive movie and TV show information effortlessly. Retrieve detailed data about films, actors, and upcoming releases using the IMDb API. Enhance your applications with rich media content and insights.',
        tools: [
            'search_imdb: Search for movies on IMDb. First 5 results are returned.\nArgs:\n    original_title: The original title of the movie to search for. Sear...',
            'get_imdb_details: Get more in depth details about a movie/series from IMDb.\nArgs:\n    imdbId: The IMDb ID of the movie/series to get details for.\nR...',
            'get_directors: Get the directors of a movie from IMDb.\nArgs:\n    imdbId: The IMDb ID of the movie to get directors for.\nReturns:\n    JSON object co...',
            'get_cast: Get the cast of a movie from IMDb.\nArgs:\n    imdbId: The IMDb ID of the movie to get cast for.\nReturns:\n    JSON object containing the ca...',
            'get_writers: Get the writers of a movie from IMDb.\nArgs:\n    imdbId: The IMDb ID of the movie to get writers for.\nReturns:\n    JSON object containi...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 83 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 83,
        title: 'Server 83/100',
        serverName: 'mcp-ollama_server',
        serverId: 'sethuram2003/mcp-ollama_server',
        description: 'Extends Model Context Protocol (MCP) to local LLMs via Ollama, enabling Claude-like tool use (files, web, email, GitHub, AI images) while keeping data private. Modular Python servers for on-prem AI. #LocalAI #MCP #Ollama',
        tools: [
            'create_calendar_event: Create, modify, and delete calendar events in Google Calendar',
            'send_meeting_invitation: Send meeting invitations to participants',
            'set_reminder: Set reminders and notifications for calendar events',
            'read_file: Read and write files securely from the file system',
            'manage_conversation_history: Manage conversation history for improved context handling'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 84 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 84,
        title: 'Server 84/100',
        serverName: 'bible-mcp',
        serverId: 'trevato/bible-mcp',
        description: 'Add biblical context to your generative AI applications.',
        tools: [
            'analyze_verse_prompt: Creates a prompt for analyzing a specific Bible verse. Parameters: reference (Bible verse reference)',
            'get_verse_by_reference: Retrieves Bible verses by reference (e.g., \'John 3:16\', \'Matthew 5:1-10\'). Parameters: reference (Bible reference string), ...',
            'get_random_verse_tool: Gets a random verse from the Bible with optional testament filtering. Parameters: translation (Translation ID, default: \'web...',
            'list_available_translations: Returns a formatted list of all available Bible translations supported by the server',
            'find_verses_on_topic_prompt: Creates a prompt for finding verses on a specific topic. Parameters: topic (topic to search for)'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 85 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 85,
        title: 'Server 85/100',
        serverName: 'scala-effect-mcp',
        serverId: 'linkyard/scala-effect-mcp',
        description: 'Library to implement model context protocol servers (MCP) in scala using fs2 and cats effect.',
        tools: [
            'ToolProvider: Functions that AI assistants can call to perform actions or retrieve information. Tools have defined input/output schemas and can be ...',
            'PromptProvider: Predefined conversation templates that can be parameterized and used to generate consistent AI responses. Prompts help standardize ...',
            'Elicitation: A mechanism for servers to request additional information from users during tool execution. This enables interactive workflows where t...',
            'RootChangeAwareProvider: Entry points that define the starting locations for resource hierarchies. Roots help organize and navigate complex data st...',
            'Logging: Built-in logging capabilities for servers to send diagnostic and informational messages to clients for debugging and monitoring. Available...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 86 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 86,
        title: 'Server 86/100',
        serverName: 'mcp',
        serverId: 'openfort-xyz/mcp',
        description: 'Connect your AI to Openfort\'s smart wallet, auth, and project infrastructure.',
        tools: [
            'authentication: Directly authenticate with Openfort from the MCP. Requires an Openfort account.',
            'initialize_project: Create new Openfort projects from the chat interface',
            'scaffold_app: Build new applications from scratch using a single prompt',
            'create_wallet: Generate wallets by instructing the LLM',
            'create_user: Generate users by instructing the LLM'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 87 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 87,
        title: 'Server 87/100',
        serverName: 'bnbchain-mcp',
        serverId: 'bnb-chain/bnbchain-mcp',
        description: 'An MCP server for interacting with BSC, opBNB, and the Greenfield blockchain.',
        tools: [
            'transfer_native_token: Transfer native tokens (BNB, ETH, MATIC, etc.) to an address. Requires PRIVATE_KEY environment variable.',
            'approve_token_spending: Approve another address to spend your ERC20 tokens. Requires PRIVATE_KEY environment variable.',
            'transfer_nft: Transfer an NFT (ERC721 token) from one address to another. Requires PRIVATE_KEY environment variable.',
            'transfer_erc1155: Transfer ERC1155 tokens to another address. Requires PRIVATE_KEY environment variable.',
            'transfer_erc20: Transfer ERC20 tokens to an address. Requires PRIVATE_KEY environment variable.'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 88 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 88,
        title: 'Server 88/100',
        serverName: 'z3_mcp',
        serverId: 'javergar/z3_mcp',
        description: 'An MCP server for the z3 theorem prover',
        tools: [
            'solve_constraint_problem: Solves a constraint satisfaction problem with a full Problem model. Takes variables (with name and type), constraints (ex...',
            'analyze_relationships: Analyzes relationships between entities with a full RelationshipQuery model. Takes relationships (person1, person2, relation...',
            'simple_constraint_solver: A simpler interface for solving constraint problems without requiring the full Problem model. Takes variables (name and t...',
            'simple_relationship_analyzer: A simpler interface for analyzing relationships without requiring the full RelationshipQuery model. Takes relationshi...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 89 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 89,
        title: 'Server 89/100',
        serverName: 'rss-mcp-server',
        serverId: 'weekend-project-space/rss-mcp-server',
        description: 'supprot fever api + remote ',
        tools: [
            'mark_item: Mark items as read/saved/unsaved using item_id and action parameters',
            'get_feeds: Retrieve all RSS feeds',
            'get_groups: Retrieve all feed groups',
            'get_unread_items: Get unread items with optional filters: max_items (default 10), max_id (default 9999999999), since_id, feed_ids, group_ids',
            'get_saved_items: Get saved/favorited items with parameters: max_items (default 10), max_id (default 9999999999)'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 90 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 90,
        title: 'Server 90/100',
        serverName: 'aidevtools',
        serverId: 'eliezedeck/aidevtools',
        description: 'A collection of tools designed to enhance AI-powered software development workflows, featuring Sidekick MCP server for process management and notifications.',
        tools: [
            'spawn_process: Start a new process with options (delay, buffer size, environment)',
            'spawn_multiple_processes: Launch multiple processes sequentially',
            'send_process_input: Send stdin input to a running process',
            'get_next_question: Register as a specialist and wait for questions',
            'answer_question: Provide an answer to a received question'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 91 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 91,
        title: 'Server 91/100',
        serverName: 'mcp-chatbot',
        serverId: 'arkcyber/mcp-chatbot',
        description: 'A powerful Rust-based chatbot MCP framework with multi-server support and tool integration capabilities.',
        tools: [
            'sqlite_operations: Database operations using SQLite for persistent data storage and querying',
            'file_operations: File system operations for reading, writing, and managing files',
            'memory_operations: Built-in support for memory operations including storage, retrieval, and management of conversation context and data',
            'rag_add: Add a new document to RAG (Retrieval Augmented Generation) database using Qdrant vector database',
            'ai_switch: Switch between AI providers (Ollama local, OpenAI, Deepseek) for model inference. Requires OpenAI API key for OpenAI provider'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 92 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 92,
        title: 'Server 92/100',
        serverName: 'docc-mcp',
        serverId: 'angelodipaolo/docc-mcp',
        description: 'MCP server for connecting AI agents to information in DocC archives',
        tools: [
            'list_archives: Lists all available DocC archives with metadata',
            'search_docc: Search across DocC documentation. Arguments: query (search term), archive (specific archive name), type (symbol type filter like struc...',
            'get_symbol: Get detailed information about a specific symbol. Arguments: symbolId (documentation path like documentation/swiftsyntax/tokensyntax), ...',
            'get_article: Get detailed information about a specific article or tutorial. Arguments: articleId (article identifier), archive (archive name)',
            'browse_archive: Browse the structure of a DocC archive. Arguments: archive (archive name), path (documentation path to browse)'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 93 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 93,
        title: 'Server 93/100',
        serverName: 'mcp-atlassian',
        serverId: 'vijay-duke/mcp-atlassian',
        description: 'MCP server for seamless Atlassian integration (Confluence & Jira) with AI assistants',
        tools: [
            'confluence_create_update_pages: Create and update pages and comments in Confluence',
            'confluence_attachments_upload: Upload attachments to Confluence pages',
            'confluence_labels_manage: Manage page labels in Confluence',
            'jira_add_comments: Add comments to issues in Jira',
            'jira_create_issue: Create new issues with custom fields in Jira'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 94 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 94,
        title: 'Server 94/100',
        serverName: 'postgres-mcp-server',
        serverId: 'satish-kori/postgres-mcp-server',
        description: 'A specialized MCP (Model Context Protocol) server for PostgreSQL that enables AI assistants to perform database operations through natural language. Built with TypeScript, featuring schema discovery, safe SQL execution, and multiple output formats.',
        tools: [
            'execute_ai_generated_sql: Execute AI-generated SQL queries with explanations. Allows execution of SQL queries generated by AI with additional conte...',
            'get_database_schema: Inspect database schemas, tables, and structure. Provides automatic discovery of all schemas, tables, and columns in the Postg...',
            'execute_sql_query: Execute SQL queries with multiple output formats (table, JSON, CSV). Includes built-in SQL injection protection and safe query e...',
            'natural_language_query: Convert natural language questions to SQL and execute them. Processes English language questions, converts them to SQL quer...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 95 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 95,
        title: 'Server 95/100',
        serverName: 'internetdata-mcp',
        serverId: 'lsd-so/internetdata-mcp',
        description: 'Updated MCP server based on TypeScript',
        tools: [
            'lsd_trip_execute: Execute custom trips (dynamic tool definitions) that extend the server\'s web interaction and data extraction capabilities',
            'lsd_research: Interactive web research and browsing capability that allows users to conduct research workflows and interact with web content',
            'lsd_extract: Structured data extraction tool that enables users to interactively select and extract repeating containers and fields of interest fro...'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 96 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 96,
        title: 'Server 96/100',
        serverName: 'Palo Alto Networks MCP Server Suite',
        serverId: '@dynamicendpoints/paloalto-mcp-server',
        description: 'Manage your Palo Alto Networks firewalls and services seamlessly through a unified API interface. Simplify firewall operations, policy management, and device monitoring with specialized servers tailored for your needs.',
        tools: [
            'multi_move_clone_configuration: Multi-Move or Multi-Clone the configuration of the Palo Alto firewall',
            'get_system_info: Get system information from the Palo Alto firewall',
            'list_resources: List resources from a specific category',
            'view_config_node_values: View configuration node values for XPath on the Palo Alto firewall'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 97 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 97,
        title: 'Server 97/100',
        serverName: 'ida-mcp-server-plugin',
        serverId: 'taida957789/ida-mcp-server-plugin',
        description: 'IDA Pro Plugin for serving MCP SSE server for cursor / claude',
        tools: [
            'define_function: Create a function at a specified address',
            'undefine_function: Undefine a function at a specified address',
            'get_bytes: Get bytes at a specified address',
            'get_disasm: Get disassembly at a specified address',
            'get_decompiled_func: Get pseudocode of the function containing the specified address'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 98 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 98,
        title: 'Server 98/100',
        serverName: 'lemon-email-mcp',
        serverId: 'manojk0303/lemon-email-mcp',
        description: 'A Model Context Protocol (MCP) server that provides email sending capabilities through the Lemon Email service. This allows AI agents and applications to send transactional emails programmatically.',
        tools: [
            'send-email: Send emails programmatically via POST request. Requires: to (recipient email), subject (email subject line), body (email content), from...',
            'health: Check system status and health of the email API service via GET request'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 99 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 99,
        title: 'Server 99/100',
        serverName: 'gitlab-clone-mcp-server',
        serverId: 'skmprb/gitlab-clone-mcp-server',
        description: 'Model Context Protocol server for GitLab API integration',
        tools: [
            'create_project: Create new GitLab projects',
            'delete_project: Delete GitLab projects',
            'update_project: Modify project name, description, visibility',
            'fork_project: Fork projects to different namespaces',
            'archive_project: Archive projects for long-term storage'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    // Server 100 - Combined page with all questions
    {
        type: 'server_combined',
        serverIndex: 100,
        title: 'Server 100/100',
        serverName: 'mcp-linux-shell-server',
        serverId: 'frost26/mcp-linux-shell-server',
        description: 'A secure Model Context Protocol (MCP) server that provides Linux shell command execution capabilities for Claude Desktop',
        tools: [
            'execute_command: Execute shell commands with proper error handling. Parameters: command (string). Commands are executed in the context of the user ...',
            'change_directory: Change working directory. Parameters: path (string). Working directory changes are isolated to the server process.',
            'get_current_directory: Get current working directory. Parameters: None. Returns the current working directory path.'
        ],
        allOnetLevels: false,
        q21Text: 'Q2.1: What is the highest autonomy level enabled by this server (choose action if one action tool)?',
        q21Choices: [
            { value: 'perception', text: 'Perception: ways a model may perceive the environment - e.g., reading files, monitoring sensors, fetching API data' },
            { value: 'reasoning', text: 'Reasoning: ways that a model may reason beyond inference - e.g., calculating results, planning actions, analyzing patterns' },
            { value: 'action', text: 'Action: ways that a model may directly affect the environment - e.g., executing commands, sending data, modifying files' }
        ],
        standardQuestions: [
            {
                questionId: 'q3',
                questionText: 'Q3: Industry Generality - Is this server cross-industry or industry-specific?',
                choices: [
            { value: '1', text: '1 - Cross-industry (e.g., desktop tools, file management, calendars)' },
            { value: '0', text: '0 - Industry-specific (e.g., crypto, payment, medical, legal systems)' }
        ]
            },
            {
                questionId: 'q4',
                questionText: 'Q4: Environment Generality - Does this server operate in open/untrusted or trusted environment?',
                choices: [
            { value: '1', text: '1 - Open/untrusted (e.g., browser automation, web scraping, generic file access)' },
            { value: '0', text: '0 - Trusted/pre-specified (e.g., specific APIs like Stripe/GitHub, internal databases)' }
        ]
            },
            {
                questionId: 'q5',
                questionText: 'Q5: Payment Autonomy Level - What is the payment autonomy level?',
                choices: [
            { value: '0', text: '0 - Not a payment server' },
            { value: '1', text: '1 - Information only (read-only payment data)' },
            { value: '2', text: '2 - Payment requests or links (e.g., payment requests)' },
            { value: '3', text: '3 - Third-party payment processing (e.g., Stripe, PayPal)' },
            { value: '4', text: '4 - Direct payment execution (e.g., crypto, credit cards)' }
        ]
            }
        ]
    },
    {
        type: 'completion',
        title: 'Thank You',
        content: `# Study Complete!

Thank you for participating. Your responses have been recorded.

Click **Finish** to complete.`
    }
];

// State
var currentPage = 0;
var responses = {};
var serverResponses = {};  // Track responses per server: {serverIndex: {question: answer}}

// Main entry point
gorilla.ready(function() {
    console.log("Study ready");
    // Initialize serverResponses
    for (var i = 0; i < studyPages.length; i++) {
        var page = studyPages[i];
        if (page.serverIndex && !serverResponses[page.serverIndex]) {
            serverResponses[page.serverIndex] = {};
        }
    }
    showPage(0);
});

// Show a page
function showPage(pageIndex) {
    currentPage = pageIndex;
    var page = studyPages[pageIndex];

    // Auto-skip func_sub pages when func_main is 'perception'
    if (page.type === 'func_sub') {
        var serverIdx = page.serverIndex;
        var funcMain = serverResponses[serverIdx] && serverResponses[serverIdx]['func_main'];

        if (funcMain === 'perception') {
            // Auto-fill with 'sensors' and skip to next page
            serverResponses[serverIdx]['func_sub'] = 'sensors';
            var responseKey = page.serverName + '_func_sub';
            responses[responseKey] = 'sensors';
            console.log('Auto-skipped func_sub for perception, set to sensors');

            // Continue to next page
            showPage(pageIndex + 1);
            return;
        }
    }

    // Clear screen
    $('#gorilla').empty();

    // Show progress
    var progress = ((pageIndex + 1) / studyPages.length) * 100;
    $('#gorilla').append(`
        <div class="progress-bar-container">
            <div class="progress-bar" style="width: ${progress}%"></div>
            <div class="progress-text">Page ${pageIndex + 1} of ${studyPages.length}</div>
        </div>
    `);

    // Show page content based on type
    if (page.type === 'completion' || page.type === 'instructions') {
        showTextPage(page);
    } else if (page.type === 'tutorial_intro') {
        showTextPage(page);
    } else if (page.type === 'tutorial_preanswered') {
        showTutorialPreansweredPage(page, pageIndex);
    } else if (page.type === 'tutorial_practice') {
        showTutorialPracticePage(page, pageIndex);
    } else if (page.type === 'tutorial_feedback') {
        showTutorialFeedbackPage(page, pageIndex);
    } else if (page.type === 'server_combined') {
        showServerCombinedPage(page, pageIndex);
    } else if (page.type === 'tool_combined') {
        showToolCombinedPage(page, pageIndex);
    } else if (page.type === 'onet_l1') {
        showONetL1Page(page, pageIndex);
    } else if (page.type === 'onet_l2') {
        showONetL2Page(page, pageIndex);
    } else if (page.type === 'onet_task') {
        showONetTaskPage(page, pageIndex);
    } else if (page.type === 'func_main') {
        showFuncMainPage(page, pageIndex);
    } else if (page.type === 'func_sub') {
        showFuncSubPage(page, pageIndex);
    } else if (page.type === 'server') {
        showServerQuestionPage(page, pageIndex);
    }

    // Show navigation
    showNavigation(pageIndex);
}

// Show text page (completion/instructions/tutorial_intro)
function showTextPage(page) {
    // Instructions and tutorial intro pages need markdown conversion
    var needsMarkdown = (page.type === 'instructions' || page.type === 'tutorial_intro');
    var content = needsMarkdown ? markdownToHtml(page.content) : page.content;

    $('#gorilla').append(`
        <div class="page-container">
            <h1>${page.title}</h1>
            <div class="content">${content}</div>
        </div>
    `);
}

// Convert markdown to HTML (basic conversion)
function markdownToHtml(markdown) {
    var html = markdown;

    // Convert headers
    html = html.replace(/^#### (.*$)/gim, '<h4>$1</h4>');
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

    // Convert bold
    html = html.replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>');

    // Convert lists
    html = html.replace(/^\* (.*$)/gim, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/gim, '<ul>$1</ul>');

    // Convert paragraphs (split by double newlines)
    var paragraphs = html.split('\\n\\n');
    html = paragraphs.map(p => {
        p = p.trim();
        // Don't wrap if already has HTML tags
        if (p.startsWith('<')) return p;
        // Replace single newlines with <br>
        p = p.replace(/\\n/g, '<br>');
        return '<p>' + p + '</p>';
    }).join('\\n');

    return html;
}

// Show server info block (reusable)
function getServerInfoHtml(page) {
    // Check if this is a tool page (has toolName) or server page (has tools array)
    if (page.toolName) {
        // Tool page
        return `
            <div class="server-info">
                <h2>Tool Information</h2>
                <p><strong>Tool Name:</strong> ${page.toolName}</p>
                <p><strong>Tool Description:</strong> ${page.toolDescription || page.description}</p>
                <p><strong>Server:</strong> ${page.serverName}</p>
                <p><strong>Server Description:</strong> ${page.serverDescription}</p>
            </div>
        `;
    } else {
        // Server page
        var toolsList = page.tools ? page.tools.map(t => `<li>${t}</li>`).join('') : '';
        return `
            <div class="server-info">
                <h2>Server Information</h2>
                <p><strong>Name:</strong> ${page.serverName}</p>
                <p><strong>Description:</strong> ${page.description}</p>
                <p><strong>Tools (sample):</strong></p>
                <ul>${toolsList}</ul>
            </div>
        `;
    }
}

// Show tutorial pre-answered page
function showTutorialPreansweredPage(page, pageIndex) {
    var answers = page.correctAnswers;
    var answersHtml = '';

    // Q1.1: O*NET Broad Category (both studies)
    answersHtml += `
        <div class="answer-block" style="background: #e8f5e9; padding: 10px; margin: 10px 0; border-left: 4px solid #4caf50;">
            <strong>Q1.1: O*NET Broad Category</strong><br>
            ✓ ${answers.onet_l1_text}
        </div>
        <div class="explanation-block" style="background: #fff3e0; padding: 10px; margin: 0 0 15px 0; border-left: 4px solid #ff9800;">
            <strong>Why this answer?</strong><br>
            ${markdownToHtml(answers.onet_l1_explanation)}
        </div>
    `;

    // Q2.1: Functionality Main (both studies)
    answersHtml += `
        <div class="answer-block" style="background: #e8f5e9; padding: 10px; margin: 10px 0; border-left: 4px solid #4caf50;">
            <strong>Q2.1: Functionality Main</strong><br>
            ✓ ${answers.func_main_text}
        </div>
        <div class="explanation-block" style="background: #fff3e0; padding: 10px; margin: 0 0 15px 0; border-left: 4px solid #ff9800;">
            <strong>Why this answer?</strong><br>
            ${markdownToHtml(answers.func_main_explanation)}
        </div>
    `;

    // Q2.2: Functionality Sub (both studies)
    answersHtml += `
        <div class="answer-block" style="background: #e8f5e9; padding: 10px; margin: 10px 0; border-left: 4px solid #4caf50;">
            <strong>Q2.2: Functionality Sub</strong><br>
            ✓ ${answers.func_sub_text}
        </div>
        <div class="explanation-block" style="background: #fff3e0; padding: 10px; margin: 0 0 15px 0; border-left: 4px solid #ff9800;">
            <strong>Why this answer?</strong><br>
            ${markdownToHtml(answers.func_sub_explanation)}
        </div>
    `;

    // Q3, Q4, Q5: Standard questions (only for servers study)
    if (answers.q3_text) {
        answersHtml += `
            <div class="answer-block" style="background: #e8f5e9; padding: 10px; margin: 10px 0; border-left: 4px solid #4caf50;">
                <strong>Q3: Industry Generality</strong><br>
                ✓ ${answers.q3_text}
            </div>
            <div class="explanation-block" style="background: #fff3e0; padding: 10px; margin: 0 0 15px 0; border-left: 4px solid #ff9800;">
                <strong>Why this answer?</strong><br>
                ${markdownToHtml(answers.q3_explanation)}
            </div>
        `;
    }

    if (answers.q4_text) {
        answersHtml += `
            <div class="answer-block" style="background: #e8f5e9; padding: 10px; margin: 10px 0; border-left: 4px solid #4caf50;">
                <strong>Q4: Environment Generality</strong><br>
                ✓ ${answers.q4_text}
            </div>
            <div class="explanation-block" style="background: #fff3e0; padding: 10px; margin: 0 0 15px 0; border-left: 4px solid #ff9800;">
                <strong>Why this answer?</strong><br>
                ${markdownToHtml(answers.q4_explanation)}
            </div>
        `;
    }

    if (answers.q5_text) {
        answersHtml += `
            <div class="answer-block" style="background: #e8f5e9; padding: 10px; margin: 10px 0; border-left: 4px solid #4caf50;">
                <strong>Q5: Payment Autonomy Level</strong><br>
                ✓ ${answers.q5_text}
            </div>
            <div class="explanation-block" style="background: #fff3e0; padding: 10px; margin: 0 0 15px 0; border-left: 4px solid #ff9800;">
                <strong>Why this answer?</strong><br>
                ${markdownToHtml(answers.q5_explanation)}
            </div>
        `;
    }

    $('#gorilla').append(`
        <div class="page-container">
            <h1>${page.title}</h1>
            ${getServerInfoHtml(page)}
            <div class="tutorial-answers">
                <h3>Correct Classifications:</h3>
                ${answersHtml}
            </div>
        </div>
    `);
}

// Show tutorial practice page
function showTutorialPracticePage(page, pageIndex) {
    $('#gorilla').append(`
        <div class="page-container">
            <h1>${page.title}</h1>
            <p style="background: #e3f2fd; padding: 10px; border-left: 4px solid #2196f3;">
                <strong>Practice Mode:</strong> Answer the questions as you normally would.
            </p>
            ${getServerInfoHtml(page)}
            <p><em>Click Next to begin classifying this server.</em></p>
        </div>
    `);
}

// Show tutorial feedback page
function showTutorialFeedbackPage(page, pageIndex) {
    var idx = page.serverIndex || page.toolIndex;  // Support both servers and tools
    var questionKey = page.questionKey;

    console.log('Feedback page - idx:', idx, 'questionKey:', questionKey);
    console.log('serverResponses[idx]:', serverResponses[idx]);

    var userAnswer = serverResponses[idx] ? serverResponses[idx][questionKey] : undefined;
    var correctValue = page.correctValue;

    console.log('userAnswer:', userAnswer, 'correctValue:', correctValue);

    var isCorrect = (userAnswer === correctValue);

    // Get text labels for ONET and functionality questions
    var userAnswerText = userAnswer;
    var correctAnswerText = correctValue;

    if (questionKey === 'onet_l1') {
        userAnswerText = onetData.l1Clusters[userAnswer] || userAnswer;
        correctAnswerText = onetData.l1Clusters[correctValue] || correctValue;
    } else if (questionKey === 'onet_l2') {
        userAnswerText = onetData.l2Clusters[userAnswer] || userAnswer;
        correctAnswerText = onetData.l2Clusters[correctValue] || correctValue;
    } else if (questionKey === 'onet_task') {
        userAnswerText = onetData.tasks[userAnswer] || userAnswer;
        correctAnswerText = onetData.tasks[correctValue] || correctValue;
    } else if (questionKey === 'func_main') {
        userAnswerText = functionalityData.mainCategories[userAnswer] || userAnswer;
        correctAnswerText = functionalityData.mainCategories[correctValue] || correctValue;
    } else if (questionKey === 'func_sub') {
        // Find subcategory text from functionalityData
        for (var mainCat in functionalityData.subCategories) {
            var subCats = functionalityData.subCategories[mainCat];
            for (var i = 0; i < subCats.length; i++) {
                if (subCats[i].value === userAnswer) {
                    userAnswerText = subCats[i].text;
                }
                if (subCats[i].value === correctValue) {
                    correctAnswerText = subCats[i].text;
                }
            }
        }
    } else if (questionKey === 'q3' || questionKey === 'q4' || questionKey === 'q5') {
        // For standard questions, find the preceding question page to get choices
        var questionPage = studyPages[pageIndex - 1];
        if (questionPage && questionPage.choices) {
            for (var j = 0; j < questionPage.choices.length; j++) {
                if (questionPage.choices[j].value === userAnswer) {
                    userAnswerText = questionPage.choices[j].text;
                }
                if (questionPage.choices[j].value === correctValue) {
                    correctAnswerText = questionPage.choices[j].text;
                }
            }
        }
    }

    var feedbackColor = isCorrect ? '#4caf50' : '#f44336';
    var feedbackBg = isCorrect ? '#e8f5e9' : '#ffebee';
    var feedbackIcon = isCorrect ? '✓' : '✗';
    var feedbackText = isCorrect ? 'Correct!' : 'Not quite right';

    var yourAnswerHtml = `<p style="background: ${feedbackBg}; padding: 10px; margin: 10px 0; border-left: 4px solid ${feedbackColor};">
               <strong>${feedbackIcon} Your answer:</strong> ${userAnswerText}
           </p>`;

    var correctAnswerHtml = isCorrect
            ? ''
            : `<p style="background: #e8f5e9; padding: 10px; margin: 10px 0; border-left: 4px solid #4caf50;">
                   <strong>✓ Correct answer:</strong> ${correctAnswerText}
               </p>`;

    $('#gorilla').append(`
        <div class="page-container">
            <h1>${page.title}</h1>
            <div style="background: ${feedbackBg}; padding: 15px; margin: 20px 0; border-left: 4px solid ${feedbackColor}; border-radius: 4px;">
                <h2 style="color: ${feedbackColor}; margin-top: 0;">${feedbackIcon} ${feedbackText}</h2>
                <h3>${page.questionTitle}</h3>
            </div>

            ${yourAnswerHtml}
            ${correctAnswerHtml}

            <div style="background: #fff3e0; padding: 15px; margin: 20px 0; border-left: 4px solid #ff9800; border-radius: 4px;">
                <h4 style="margin-top: 0;">💡 Tip:</h4>
                <p>${page.feedbackTip}</p>
            </div>

            <p style="text-align: center; margin-top: 30px;">
                <em>Click Next to continue with the practice questions.</em>
            </p>
        </div>
    `);
}

// Show O*NET Level 1 page
function showONetL1Page(page, pageIndex) {
    var choices = [];
    for (var l1Id in onetData.l1Clusters) {
        choices.push({
            value: l1Id,
            text: onetData.l1Clusters[l1Id]
        });
    }

    var choicesHtml = choices.map(choice => `
        <label>
            <input type="radio" name="question_${pageIndex}" value="${choice.value}">
            ${choice.text}
        </label>
    `).join('');

    $('#gorilla').append(`
        <div class="page-container">
            <h1>${page.title}</h1>
            ${getServerInfoHtml(page)}
            <div class="question-block">
                <p class="question-text">${page.questionText} <span class="required">*</span></p>
                ${choicesHtml}
            </div>
        </div>
    `);
}

// Show O*NET Level 2 page (conditional on L1)
function showONetL2Page(page, pageIndex) {
    // Get previous L1 selection
    var serverIdx = page.serverIndex;
    var l1Selection = serverResponses[serverIdx]['onet_l1'];

    if (!l1Selection) {
        $('#gorilla').append(`
            <div class="page-container">
                <h1>Error</h1>
                <p>Missing L1 selection. Please go back.</p>
            </div>
        `);
        return;
    }

    var choices = getL2ClustersForL1(l1Selection);

    var choicesHtml = choices.map(choice => `
        <label>
            <input type="radio" name="question_${pageIndex}" value="${choice.value}">
            ${choice.text}
        </label>
    `).join('');

    $('#gorilla').append(`
        <div class="page-container">
            <h1>${page.title}</h1>
            ${getServerInfoHtml(page)}
            <div class="question-block">
                <p class="question-text">${page.questionText} <span class="required">*</span></p>
                <p><em>Based on your L1 selection: ${onetData.l1Clusters[l1Selection]}</em></p>
                ${choicesHtml}
            </div>
        </div>
    `);
}

// Show O*NET Task page (conditional on L2)
function showONetTaskPage(page, pageIndex) {
    // Get previous L2 selection
    var serverIdx = page.serverIndex;
    var l2Selection = serverResponses[serverIdx]['onet_l2'];

    if (!l2Selection) {
        $('#gorilla').append(`
            <div class="page-container">
                <h1>Error</h1>
                <p>Missing L2 selection. Please go back.</p>
            </div>
        `);
        return;
    }

    var choices = getTasksForL2(l2Selection);

    var choicesHtml = choices.map(choice => `
        <label>
            <input type="radio" name="question_${pageIndex}" value="${choice.value}">
            ${choice.text}
        </label>
    `).join('');

    $('#gorilla').append(`
        <div class="page-container">
            <h1>${page.title}</h1>
            ${getServerInfoHtml(page)}
            <div class="question-block">
                <p class="question-text">${page.questionText} <span class="required">*</span></p>
                <p><em>Based on your L2 selection: ${onetData.l2Clusters[l2Selection]}</em></p>
                ${choicesHtml}
            </div>
        </div>
    `);
}

// Show functionality main category page
function showFuncMainPage(page, pageIndex) {
    var choicesHtml = page.choices.map(choice => `
        <label>
            <input type="radio" name="question_${pageIndex}" value="${choice.value}">
            ${choice.text}
        </label>
    `).join('');

    $('#gorilla').append(`
        <div class="page-container">
            <h1>${page.title}</h1>
            ${getServerInfoHtml(page)}
            <div class="question-block">
                <p class="question-text">${page.questionText} <span class="required">*</span></p>
                ${choicesHtml}
            </div>
        </div>
    `);
}

// Show functionality sub-category page (conditional on main)
function showFuncSubPage(page, pageIndex) {
    // Get previous main category selection
    var idx = page.serverIndex || page.toolIndex;  // Support both servers and tools
    var mainSelection = serverResponses[idx] ? serverResponses[idx]['func_main'] : undefined;

    if (!mainSelection) {
        $('#gorilla').append(`
            <div class="page-container">
                <h1>Error</h1>
                <p>Missing main functionality selection. Please go back.</p>
            </div>
        `);
        return;
    }

    var choices = getSubCategoriesFor(mainSelection);

    var choicesHtml = choices.map(choice => `
        <label>
            <input type="radio" name="question_${pageIndex}" value="${choice.value}">
            ${choice.text}
        </label>
    `).join('');

    $('#gorilla').append(`
        <div class="page-container">
            <h1>${page.title}</h1>
            ${getServerInfoHtml(page)}
            <div class="question-block">
                <p class="question-text">${page.questionText} <span class="required">*</span></p>
                <p><em>Based on your main category: ${functionalityData.mainCategories[mainSelection]}</em></p>
                ${choicesHtml}
            </div>
        </div>
    `);
}

// Show combined server page with all questions
function showServerCombinedPage(page, pageIndex) {
    var serverIdx = page.serverIndex;
    var toolsList = page.tools.map(t => `<li>${t}</li>`).join('');

    // Build server info section
    var serverInfoHtml = `
        <div class="server-info">
            <h2>Server Information</h2>
            <p><strong>Name:</strong> ${page.serverName}</p>
            <p><strong>Description:</strong> ${page.description}</p>
            <p><strong>Tools (sample):</strong></p>
            <ul>${toolsList}</ul>
        </div>
    `;

    // Q1.1: O*NET Level 1 (always visible)
    var onetL1Choices = [];
    for (var l1Id in onetData.l1Clusters) {
        onetL1Choices.push({value: l1Id, text: onetData.l1Clusters[l1Id]});
    }
    var onetL1Html = onetL1Choices.map(choice => `
        <label>
            <input type="radio" name="onet_l1_${serverIdx}" value="${choice.value}" class="onet-l1-radio">
            ${choice.text}
        </label>
    `).join('');

    var q1Html = `
        <div class="question-block" id="q1_block">
            <h3>Q1.1: O*NET Broad Category</h3>
            <p class="question-text">Which broad occupational category best describes the primary function of this server? <span class="required">*</span></p>
            ${onetL1Html}
        </div>
    `;

    // Q1.2 and Q1.3: O*NET Level 2 and Task (conditionally visible)
    var q12Html = '';
    var q13Html = '';
    if (page.allOnetLevels) {
        q12Html = `
            <div class="question-block conditional-question" id="q12_block" style="display: none; margin-left: 20px; padding-left: 20px; border-left: 3px solid #ccc;">
                <h3>Q1.2: O*NET Sub-Category</h3>
                <p class="question-text">Which specific occupational sub-category best fits this server? <span class="required">*</span></p>
                <div id="onet_l2_choices_${serverIdx}"></div>
            </div>
        `;

        q13Html = `
            <div class="question-block conditional-question" id="q13_block" style="display: none; margin-left: 40px; padding-left: 20px; border-left: 3px solid #ccc;">
                <h3>Q1.3: O*NET Task</h3>
                <p class="question-text">Which specific occupational task most closely matches this server's functionality? <span class="required">*</span></p>
                <div id="onet_task_choices_${serverIdx}"></div>
            </div>
        `;
    }

    // Q2.1: Functionality Main Category (always visible)
    var q21Html = page.q21Choices.map(choice => `
        <label>
            <input type="radio" name="q21_${serverIdx}" value="${choice.value}" class="q21-radio">
            ${choice.text}
        </label>
    `).join('');

    var q2Html = `
        <div class="question-block" id="q2_block">
            <h3>Q2.1: Autonomy Level</h3>
            <p class="question-text">${page.q21Text} <span class="required">*</span></p>
            ${q21Html}
        </div>
    `;

    // Q2.2: Functionality Sub-Category (conditionally visible)
    var q22Html = `
        <div class="question-block conditional-question" id="q22_block" style="display: none; margin-left: 20px; padding-left: 20px; border-left: 3px solid #ccc;">
            <h3>Q2.2: Sub-Category</h3>
            <p class="question-text">Which specific sub-category best describes this server's functionality? <span class="required">*</span></p>
            <div id="q22_choices_${serverIdx}"></div>
        </div>
    `;

    // Standard questions (Q3, Q4, Q5)
    var standardQuestionsHtml = page.standardQuestions.map(function(q, idx) {
        var qNum = idx + 3;
        var choicesHtml = q.choices.map(choice => `
            <label>
                <input type="radio" name="${q.questionId}_${serverIdx}" value="${choice.value}">
                ${choice.text}
            </label>
        `).join('');

        return `
            <div class="question-block" id="q${qNum}_block">
                <h3>${q.questionId.toUpperCase()}</h3>
                <p class="question-text">${q.questionText} <span class="required">*</span></p>
                ${choicesHtml}
            </div>
        `;
    }).join('');

    // Assemble the full page
    $('#gorilla').append(`
        <div class="page-container">
            <h1>${page.title}</h1>
            ${serverInfoHtml}
            ${q1Html}
            ${q12Html}
            ${q13Html}
            ${q2Html}
            ${q22Html}
            ${standardQuestionsHtml}
        </div>
    `);

    // Add event listeners for conditional display

    // O*NET L1 -> L2 conditional display
    if (page.allOnetLevels) {
        $(`.onet-l1-radio`).on('change', function() {
            var selectedL1 = $('input[name="onet_l1_' + serverIdx + '"]:checked').val();
            if (selectedL1) {
                // Show L2 block and populate choices
                $('#q12_block').show();
                var l2Choices = getL2ClustersForL1(selectedL1);
                var l2Html = l2Choices.map(choice => `
                    <label>
                        <input type="radio" name="onet_l2_${serverIdx}" value="${choice.value}" class="onet-l2-radio">
                        ${choice.text}
                    </label>
                `).join('');
                $('#onet_l2_choices_' + serverIdx).html(l2Html);

                // Add listener for L2 -> Task
                $(`.onet-l2-radio`).on('change', function() {
                    var selectedL2 = $('input[name="onet_l2_' + serverIdx + '"]:checked').val();
                    if (selectedL2) {
                        // Show Task block and populate choices
                        $('#q13_block').show();
                        var taskChoices = getTasksForL2(selectedL2);
                        var taskHtml = taskChoices.map(choice => `
                            <label>
                                <input type="radio" name="onet_task_${serverIdx}" value="${choice.value}">
                                ${choice.text}
                            </label>
                        `).join('');
                        $('#onet_task_choices_' + serverIdx).html(taskHtml);
                    }
                });
            }
        });
    }

    // Q2.1 -> Q2.2 conditional display
    $(`.q21-radio`).on('change', function() {
        var selectedQ21 = $('input[name="q21_' + serverIdx + '"]:checked').val();
        if (selectedQ21) {
            // Auto-fill perception with sensors
            if (selectedQ21 === 'perception') {
                $('#q22_block').hide();
                // Auto-set response
                if (!serverResponses[serverIdx]) serverResponses[serverIdx] = {};
                serverResponses[serverIdx]['func_sub'] = 'sensors';
                responses[page.serverName + '_func_sub'] = 'sensors';
            } else {
                // Show Q2.2 block and populate choices
                $('#q22_block').show();
                var subChoices = getSubCategoriesFor(selectedQ21);
                var subHtml = subChoices.map(choice => `
                    <label>
                        <input type="radio" name="q22_${serverIdx}" value="${choice.value}">
                        ${choice.text}
                    </label>
                `).join('');
                $('#q22_choices_' + serverIdx).html(subHtml);
            }
        }
    });
}

// Show combined tool page with all questions (Q1.1, Q2.1, Q2.2)
function showToolCombinedPage(page, pageIndex) {
    var toolIdx = page.toolIndex;

    // Build tool info section
    var toolInfoHtml = `
        <div class="server-info">
            <h2>Tool Information</h2>
            <p><strong>Tool Name:</strong> ${page.toolName}</p>
            <p><strong>Tool Description:</strong> ${page.toolDescription}</p>
            <p><strong>Server:</strong> ${page.serverName}</p>
            <p><strong>Server Description:</strong> ${page.serverDescription}</p>
        </div>
    `;

    // Q1.1: O*NET Level 1 (always visible)
    var onetL1Choices = [];
    for (var l1Id in onetData.l1Clusters) {
        onetL1Choices.push({value: l1Id, text: onetData.l1Clusters[l1Id]});
    }
    var onetL1Html = onetL1Choices.map(choice => `
        <label>
            <input type="radio" name="onet_l1_${toolIdx}" value="${choice.value}">
            ${choice.text}
        </label>
    `).join('');

    var q1Html = `
        <div class="question-block" id="q1_block">
            <h3>Q1.1: O*NET Broad Category</h3>
            <p class="question-text">Which broad occupational category best describes what this tool does? <span class="required">*</span></p>
            ${onetL1Html}
        </div>
    `;

    // Q2.1: Functionality Main Category (always visible)
    var q21Html = page.q21Choices.map(choice => `
        <label>
            <input type="radio" name="q21_${toolIdx}" value="${choice.value}" class="q21-radio">
            ${choice.text}
        </label>
    `).join('');

    var q2Html = `
        <div class="question-block" id="q2_block">
            <h3>Q2.1: Autonomy Level</h3>
            <p class="question-text">${page.q21Text} <span class="required">*</span></p>
            ${q21Html}
        </div>
    `;

    // Q2.2: Functionality Sub-Category (conditionally visible)
    var q22Html = `
        <div class="question-block conditional-question" id="q22_block" style="display: none; margin-left: 20px; padding-left: 20px; border-left: 3px solid #ccc;">
            <h3>Q2.2: Sub-Category</h3>
            <p class="question-text">Which specific sub-category best describes this tool's functionality? <span class="required">*</span></p>
            <div id="q22_choices_${toolIdx}"></div>
        </div>
    `;

    // Assemble the full page
    $('#gorilla').append(`
        <div class="page-container">
            <h1>${page.title}</h1>
            ${toolInfoHtml}
            ${q1Html}
            ${q2Html}
            ${q22Html}
        </div>
    `);

    // Add event listeners for conditional display

    // Q2.1 -> Q2.2 conditional display
    $(`.q21-radio`).on('change', function() {
        var selectedQ21 = $('input[name="q21_' + toolIdx + '"]:checked').val();
        if (selectedQ21) {
            // Auto-fill perception with sensors
            if (selectedQ21 === 'perception') {
                $('#q22_block').hide();
                // Auto-set response
                if (!serverResponses[toolIdx]) serverResponses[toolIdx] = {};
                serverResponses[toolIdx]['func_sub'] = 'sensors';
                responses[page.toolName + '_func_sub'] = 'sensors';
            } else {
                // Show Q2.2 block and populate choices
                $('#q22_block').show();
                var subChoices = getSubCategoriesFor(selectedQ21);
                var subHtml = subChoices.map(choice => `
                    <label>
                        <input type="radio" name="q22_${toolIdx}" value="${choice.value}">
                        ${choice.text}
                    </label>
                `).join('');
                $('#q22_choices_' + toolIdx).html(subHtml);
            }
        }
    });
}

// Show standard server question page
function showServerQuestionPage(page, pageIndex) {
    var toolsList = page.tools.map(t => `<li>${t}</li>`).join('');

    var choicesHtml = page.choices.map(choice => `
        <label>
            <input type="radio" name="question_${pageIndex}" value="${choice.value}">
            ${choice.text}
        </label>
    `).join('');

    $('#gorilla').append(`
        <div class="page-container">
            <h1>${page.title}</h1>
            ${getServerInfoHtml(page)}
            <div class="question-block">
                <p class="question-text">${page.questionText} <span class="required">*</span></p>
                ${choicesHtml}
            </div>
        </div>
    `);
}

// Show navigation buttons
function showNavigation(pageIndex) {
    var isFirst = pageIndex === 0;
    var isLast = pageIndex === studyPages.length - 1;

    var backButton = isFirst ? '' : '<button id="back-btn" class="btn">Back</button>';
    var nextText = isLast ? 'Finish' : 'Next';

    $('#gorilla').append(`
        <div class="navigation">
            ${backButton}
            <button id="next-btn" class="btn btn-primary">${nextText}</button>
        </div>
    `);

    // Back button
    if (!isFirst) {
        $('#back-btn').on('click', function() {
            var targetPage = currentPage - 1;

            // Skip backwards over auto-skipped func_sub pages
            if (targetPage >= 0) {
                var prevPage = studyPages[targetPage];
                if (prevPage.type === 'func_sub') {
                    var serverIdx = prevPage.serverIndex;
                    var funcMain = serverResponses[serverIdx] && serverResponses[serverIdx]['func_main'];
                    if (funcMain === 'perception') {
                        // Skip one more page back to func_main
                        targetPage = targetPage - 1;
                    }
                }
            }

            showPage(targetPage);
        });
    }

    // Next button
    $('#next-btn').on('click', function() {
        if (validatePage()) {
            if (isLast) {
                finishStudy();
            } else {
                showPage(currentPage + 1);
            }
        }
    });
}

// Validate current page
function validatePage() {
    var page = studyPages[currentPage];

    // Text pages don't need validation
    if (page.type === 'completion' || page.type === 'instructions') {
        return true;
    }

    // Tutorial pages don't need validation
    if (page.type === 'tutorial_intro' || page.type === 'tutorial_preanswered' || page.type === 'tutorial_practice' || page.type === 'tutorial_feedback') {
        return true;
    }

    // Combined tool page validation
    if (page.type === 'tool_combined') {
        var toolIdx = page.toolIndex;
        var isValid = true;
        var missingFields = [];

        // Q1.1: O*NET L1
        var onetL1 = $('input[name="onet_l1_' + toolIdx + '"]:checked').val();
        if (!onetL1) {
            missingFields.push('Q1.1: O*NET Broad Category');
            isValid = false;
        } else {
            serverResponses[toolIdx]['onet_l1'] = onetL1;
            responses[page.toolName + '_onet_l1'] = onetL1;
        }

        // Q2.1: Functionality Main
        var q21 = $('input[name="q21_' + toolIdx + '"]:checked').val();
        if (!q21) {
            missingFields.push('Q2.1: Autonomy Level');
            isValid = false;
        } else {
            serverResponses[toolIdx]['func_main'] = q21;
            responses[page.toolName + '_func_main'] = q21;

            // Q2.2: Functionality Sub (not required for perception - auto-filled)
            if (q21 !== 'perception' && $('#q22_block').is(':visible')) {
                var q22 = $('input[name="q22_' + toolIdx + '"]:checked').val();
                if (!q22) {
                    missingFields.push('Q2.2: Sub-Category');
                    isValid = false;
                } else {
                    serverResponses[toolIdx]['func_sub'] = q22;
                    responses[page.toolName + '_func_sub'] = q22;
                }
            }
        }

        if (!isValid) {
            alert('Please answer all required questions:\n\n' + missingFields.join('\n'));
            return false;
        }

        return true;
    }

    // Combined server page validation
    if (page.type === 'server_combined') {
        var serverIdx = page.serverIndex;
        var isValid = true;
        var missingFields = [];

        // Q1.1: O*NET L1
        var onetL1 = $('input[name="onet_l1_' + serverIdx + '"]:checked').val();
        if (!onetL1) {
            missingFields.push('Q1.1: O*NET Broad Category');
            isValid = false;
        } else {
            serverResponses[serverIdx]['onet_l1'] = onetL1;
            responses[page.serverName + '_onet_l1'] = onetL1;
        }

        // Q1.2 and Q1.3 (if allOnetLevels is true)
        if (page.allOnetLevels) {
            // Q1.2: O*NET L2 (only required if L1 is selected and L2 block is visible)
            if (onetL1 && $('#q12_block').is(':visible')) {
                var onetL2 = $('input[name="onet_l2_' + serverIdx + '"]:checked').val();
                if (!onetL2) {
                    missingFields.push('Q1.2: O*NET Sub-Category');
                    isValid = false;
                } else {
                    serverResponses[serverIdx]['onet_l2'] = onetL2;
                    responses[page.serverName + '_onet_l2'] = onetL2;

                    // Q1.3: O*NET Task (only required if L2 is selected and task block is visible)
                    if ($('#q13_block').is(':visible')) {
                        var onetTask = $('input[name="onet_task_' + serverIdx + '"]:checked').val();
                        if (!onetTask) {
                            missingFields.push('Q1.3: O*NET Task');
                            isValid = false;
                        } else {
                            serverResponses[serverIdx]['onet_task'] = onetTask;
                            responses[page.serverName + '_onet_task'] = onetTask;
                        }
                    }
                }
            }
        }

        // Q2.1: Functionality Main
        var q21 = $('input[name="q21_' + serverIdx + '"]:checked').val();
        if (!q21) {
            missingFields.push('Q2.1: Autonomy Level');
            isValid = false;
        } else {
            serverResponses[serverIdx]['func_main'] = q21;
            responses[page.serverName + '_func_main'] = q21;

            // Q2.2: Functionality Sub (not required for perception - auto-filled)
            if (q21 !== 'perception' && $('#q22_block').is(':visible')) {
                var q22 = $('input[name="q22_' + serverIdx + '"]:checked').val();
                if (!q22) {
                    missingFields.push('Q2.2: Sub-Category');
                    isValid = false;
                } else {
                    serverResponses[serverIdx]['func_sub'] = q22;
                    responses[page.serverName + '_func_sub'] = q22;
                }
            }
        }

        // Standard questions (Q3, Q4, Q5, etc.)
        for (var i = 0; i < page.standardQuestions.length; i++) {
            var q = page.standardQuestions[i];
            var qAnswer = $('input[name="' + q.questionId + '_' + serverIdx + '"]:checked').val();
            if (!qAnswer) {
                missingFields.push(q.questionId.toUpperCase());
                isValid = false;
            } else {
                serverResponses[serverIdx][q.questionId] = qAnswer;
                responses[page.serverName + '_' + q.questionId] = qAnswer;
            }
        }

        if (!isValid) {
            alert('Please answer all required questions:\n\n' + missingFields.join('\n'));
            return false;
        }

        return true;
    }

    // All other question pages
    var answer = $('input[name="question_' + currentPage + '"]:checked').val();

    // For practice pages, allow skipping but save answer if provided
    if (!answer && !page.isPractice) {
        alert('Please answer the question');
        return false;
    }

    // Store response if an answer was provided
    if (answer) {
        var idx = page.serverIndex || page.toolIndex;  // Support both servers and tools
        if (idx) {
            // Initialize if needed
            if (!serverResponses[idx]) {
                serverResponses[idx] = {};
                console.log('Initialized serverResponses[' + idx + ']');
            }

            // Store in server/tool-specific responses
            serverResponses[idx][page.question] = answer;
            console.log('Stored answer: serverResponses[' + idx + '][' + page.question + '] = ' + answer);

            // Also store in global responses with server/tool name prefix
            var nameKey = page.serverName || page.toolName || 'unknown';
            var responseKey = nameKey + '_' + page.question;
            responses[responseKey] = answer;
        } else {
            responses[page.question] = answer;
        }
    } else if (page.isPractice) {
        // For practice pages, still initialize the response object even if no answer
        var idx = page.serverIndex || page.toolIndex;
        if (idx && !serverResponses[idx]) {
            serverResponses[idx] = {};
            console.log('Initialized serverResponses[' + idx + '] for practice page (no answer yet)');
        }
    }

    return true;
}

// Finish study
function finishStudy() {
    console.log("Final responses:", responses);
    console.log("Server responses:", serverResponses);

    // Upload metrics
    for (var key in responses) {
        // Extract server name and question ID
        var lastUnderscoreQIndex = key.lastIndexOf('_q');
        var lastUnderscoreOIndex = key.lastIndexOf('_onet');
        var lastUnderscoreFIndex = key.lastIndexOf('_func');

        var serverName = key;
        var questionId = '';

        if (lastUnderscoreQIndex !== -1) {
            serverName = key.substring(0, lastUnderscoreQIndex);
            questionId = key.substring(lastUnderscoreQIndex + 1);
        } else if (lastUnderscoreOIndex !== -1) {
            serverName = key.substring(0, lastUnderscoreOIndex);
            questionId = key.substring(lastUnderscoreOIndex + 1);
        } else if (lastUnderscoreFIndex !== -1) {
            serverName = key.substring(0, lastUnderscoreFIndex);
            questionId = key.substring(lastUnderscoreFIndex + 1);
        }

        gorilla.metric({
            name: key,
            value: responses[key],
            checked: '1',
            servername: serverName,
            question: questionId
        });
    }

    gorilla.finish();
}
