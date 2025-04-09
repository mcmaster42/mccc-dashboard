import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import numpy as np

# Set page config
st.set_page_config(
    page_title="MCCC Program Analysis Dashboard",
    page_icon="🏫",
    layout="wide"
)

# Load the data
@st.cache_data
def load_data():
    # In a real application, you would load from a file
    # For the sake of this example, we'll create the dataframe from the provided data
    data = {
        "program_name": ["Software Engineering", "Welding Technology", "Cosmetology", "Criminal Justice", "Automotive Technology", 
                       "Health Careers", "Digital Design", "Construction Trades", "Nursing Assistant", "Dental Assisting", 
                       "Electrical Technician", "Medical Billing", "Practical Nursing", "HVAC Certification", "Industrial Maintenance", 
                       "College Credit Plus", "Honors English", "Applied Mathematics", "Advanced Science", "CPR Certification", 
                       "Firefighter Safety", "OSHA 10-Hour Training", "Forklift Training", "ServSafe Manager", 
                       "State Cosmetology Exam Prep", "CompTIA A+ Cert", "STNA Certification", "Resume Writing Workshop", 
                       "Job Interview Skills", "Professional Etiquette Training"],
        "program_type": ["High School Career Tech", "High School Career Tech", "High School Career Tech", "High School Career Tech", 
                       "High School Career Tech", "High School Career Tech", "High School Career Tech", "High School Career Tech", 
                       "Adult Education", "Adult Education", "Adult Education", "Adult Education", "Adult Education", 
                       "Adult Education", "Adult Education", "Academic Programs", "Academic Programs", "Academic Programs", 
                       "Academic Programs", "Continuing Education", "Continuing Education", "Continuing Education", 
                       "Continuing Education", "Credentialing & Certification", "Credentialing & Certification", 
                       "Credentialing & Certification", "Credentialing & Certification", "Workforce Development", 
                       "Workforce Development", "Workforce Development"],
        "total_students": [58, 44, 23, 44, 28, 48, 57, 28, 55, 36, 26, 35, 29, 34, 32, 40, 37, 20, 23, 21, 19, 28, 60, 35, 60, 33, 40, 44, 47, 32],
        "boys": [8, 0, 14, 20, 28, 17, 44, 0, 37, 2, 7, 19, 11, 17, 27, 39, 36, 14, 18, 8, 13, 5, 35, 1, 21, 29, 33, 30, 6, 27],
        "girls": [50, 44, 9, 24, 0, 31, 13, 28, 18, 34, 19, 16, 18, 17, 5, 1, 1, 6, 5, 13, 6, 23, 25, 34, 39, 4, 7, 14, 41, 5],
        "districts": ["Wadsworth, Cloverleaf", "Cloverleaf, Brunswick", "Buckeye, Brunswick, Medina", "Cloverleaf, Highland, Medina", 
                    "Highland, Medina, Wadsworth", "Highland", "Buckeye", "Highland, Wadsworth, Buckeye", "Buckeye, Wadsworth", 
                    "Cloverleaf, Medina, Highland", "Wadsworth, Brunswick", "Wadsworth, Buckeye, Medina", "Highland", 
                    "Wadsworth", "Brunswick, Wadsworth", "Medina", "Medina", "Buckeye, Highland, Cloverleaf", 
                    "Buckeye, Cloverleaf", "Buckeye", "Buckeye, Highland, Medina", "Highland, Cloverleaf", "Wadsworth, Brunswick", 
                    "Cloverleaf, Medina", "Medina, Cloverleaf, Brunswick", "Wadsworth, Brunswick", "Highland, Medina, Wadsworth", 
                    "Buckeye, Cloverleaf", "Buckeye", "Highland"],
        "instructor": ["Mr. Patel", "Ms. Rivera", "Mr. Smith", "Mrs. Gaines", "Ms. Johnson", "Mr. Thompson", "Ms. Rivera", 
                      "Ms. Rivera", "Mrs. Lee", "Ms. Johnson", "Ms. Johnson", "Mr. Patel", "Ms. Rivera", "Mr. Thompson", 
                      "Mr. Patel", "Mr. Patel", "Mr. Thompson", "Mrs. Gaines", "Ms. Rivera", "Mrs. Gaines", "Mr. Patel", 
                      "Ms. Rivera", "Mr. Thompson", "Ms. Johnson", "Mr. Thompson", "Ms. Johnson", "Mrs. Lee", "Ms. Johnson", 
                      "Mrs. Gaines", "Ms. Rivera"],
        "credentials": ["No", "No", "No", "Yes", "No", "Yes", "No", "No", "Yes", "No", "No", "Yes", "Yes", "No", "Yes", "Yes", 
                      "No", "No", "No", "Yes", "Yes", "No", "No", "Yes", "No", "No", "Yes", "Yes", "Yes", "Yes"],
        "job_placement_rate": [0.93, 0.88, 0.93, 0.93, 0.71, 0.84, 0.90, 0.78, 0.80, 0.73, 0.82, 0.77, 0.75, 0.76, 0.92, 0.90, 
                              0.83, 0.90, 0.75, 0.82, 0.81, 0.94, 0.83, 0.85, 0.93, 0.75, 0.78, 0.84, 0.89, 0.87],
        "certifications": ["Python, Java, Web Dev", "AWS Cert, OSHA10", "State License", "None", "CPR", "Forklift Safety", 
                         "ServSafe", "Forklift Safety", "STNA, CPR", "None", "None", "CompTIA A+", "STNA, CPR", "ServSafe", 
                         "Forklift Safety", "CompTIA A+", "Forklift Safety", "ServSafe", "CompTIA A+", "Forklift Safety", 
                         "Firefighter Level 1", "CompTIA A+", "ServSafe", "CompTIA A+", "State License", "ServSafe", 
                         "STNA, CPR", "None", "CompTIA A+", "None"],
        "room": [303, 356, 177, 146, 145, 165, 375, 293, 170, 377, 392, 207, 162, 265, 133, 159, 226, 146, 189, 333, 147, 278, 
               310, 204, 328, 318, 312, 237, 365, 111]
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Extract districts into a list for each program
    df['districts_list'] = df['districts'].str.split(', ')
    
    # Create additional metrics
    df['gender_ratio'] = df['boys'] / df['total_students']
    df['credentials_binary'] = df['credentials'].map({'Yes': 1, 'No': 0})
    
    return df

# Load the data
df = load_data()

# Process district data for analysis
def process_district_data(df):
    all_districts = []
    for district_list in df['districts_list']:
        all_districts.extend(district_list)
    
    district_counts = pd.Series(all_districts).value_counts().reset_index()
    district_counts.columns = ['District', 'Count']
    return district_counts

district_counts = process_district_data(df)

# Dashboard Header
st.title("🏫 Medina County Career Center Program Analytics")
st.markdown("""
This dashboard provides an interactive analysis of programs offered at the Medina County Career Center, 
their enrollment demographics, and performance metrics. Use the filters in the sidebar to explore specific aspects of the data.
""")

# Sidebar filters
st.sidebar.header("Filters")
program_types = ["All"] + sorted(df['program_type'].unique().tolist())
selected_program_type = st.sidebar.selectbox("Program Type", program_types)

districts = ["All"] + sorted(set([district for sublist in df['districts_list'] for district in sublist]))
selected_district = st.sidebar.selectbox("School District", districts)

# Filter the data based on selections
filtered_df = df.copy()
if selected_program_type != "All":
    filtered_df = filtered_df[filtered_df['program_type'] == selected_program_type]
    
if selected_district != "All":
    mask = filtered_df['districts'].str.contains(selected_district)
    filtered_df = filtered_df[mask]

# Display key metrics
st.header("Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Programs", len(filtered_df))
    
with col2:
    st.metric("Total Students", filtered_df['total_students'].sum())
    
with col3:
    avg_placement = filtered_df['job_placement_rate'].mean()
    st.metric("Average Placement Rate", f"{avg_placement:.1%}")
    
with col4:
    credentialed_percent = filtered_df['credentials_binary'].mean() * 100
    st.metric("Programs with Credentialed Instructors", f"{credentialed_percent:.1f}%")

# Program Type Distribution
st.header("Program Distribution Analysis")
tab1, tab2 = st.tabs(["Program Types", "Enrollment Demographics"])

with tab1:
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Program Type Distribution
        program_type_counts = filtered_df['program_type'].value_counts().reset_index()
        program_type_counts.columns = ['Program Type', 'Count']
        
        fig_program_types = px.pie(
            program_type_counts, 
            values='Count', 
            names='Program Type', 
            title="Distribution of Program Types",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_program_types.update_layout(
            legend=dict(orientation="h", y=-0.1)
        )
        st.plotly_chart(fig_program_types, use_container_width=True)
    
    with col2:
        # Top programs by enrollment
        top_programs = filtered_df.sort_values('total_students', ascending=False).head(10)
        fig_top_programs = px.bar(
            top_programs,
            x='total_students',
            y='program_name',
            color='program_type',
            orientation='h',
            title="Top 10 Programs by Enrollment",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_top_programs.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Total Students",
            yaxis_title="Program Name",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_top_programs, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        # Gender distribution by program type
        gender_by_program = filtered_df.groupby('program_type').agg({
            'boys': 'sum',
            'girls': 'sum'
        }).reset_index()
        
        gender_by_program_melted = pd.melt(
            gender_by_program, 
            id_vars=['program_type'],
            value_vars=['boys', 'girls'],
            var_name='Gender',
            value_name='Count'
        )
        
        fig_gender = px.bar(
            gender_by_program_melted,
            x='program_type',
            y='Count',
            color='Gender',
            barmode='group',
            title="Gender Distribution by Program Type",
            color_discrete_map={'boys': '#3498db', 'girls': '#e74c3c'}
        )
        fig_gender.update_layout(
            xaxis_title="Program Type",
            yaxis_title="Number of Students",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_gender, use_container_width=True)
    
    with col2:
        # Programs with extreme gender ratios
        df_sorted_by_ratio = filtered_df.copy()
        df_sorted_by_ratio['boys_percent'] = df_sorted_by_ratio['boys'] / df_sorted_by_ratio['total_students'] * 100
        
        # Get top 5 male-dominated and top 5 female-dominated programs
        male_dominated = df_sorted_by_ratio.nlargest(5, 'boys_percent')
        female_dominated = df_sorted_by_ratio.nsmallest(5, 'boys_percent')
        
        # Combine them
        extreme_ratios = pd.concat([male_dominated, female_dominated])
        extreme_ratios['gender_balance'] = extreme_ratios.apply(
            lambda x: f"{x['boys_percent']:.0f}% Male / {100 - x['boys_percent']:.0f}% Female", axis=1
        )
        
        fig_extreme = px.bar(
            extreme_ratios,
            x='program_name',
            y=['boys', 'girls'],
            title="Programs with Most Extreme Gender Ratios",
            barmode='stack',
            color_discrete_map={'boys': '#3498db', 'girls': '#e74c3c'},
            text=extreme_ratios['gender_balance']
        )
        fig_extreme.update_layout(
            xaxis_title="Program",
            yaxis_title="Number of Students",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis={'categoryorder': 'total ascending'}
        )
        fig_extreme.update_traces(textposition='inside')
        st.plotly_chart(fig_extreme, use_container_width=True)

# District Analysis
st.header("School District Analysis")
tab1, tab2 = st.tabs(["District Participation", "Program Distribution"])

with tab1:
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Process district data for the filtered dataset
        if selected_district == "All" and selected_program_type == "All":
            current_district_counts = district_counts
        else:
            current_district_counts = process_district_data(filtered_df)
        
        fig_districts = px.bar(
            current_district_counts,
            x='District',
            y='Count',
            title="Program Participation by School District",
            color='Count',
            color_continuous_scale=px.colors.sequential.Viridis
        )
        fig_districts.update_layout(
            xaxis_title="School District",
            yaxis_title="Number of Programs",
            xaxis={'categoryorder': 'total descending'}
        )
        st.plotly_chart(fig_districts, use_container_width=True)
    
    with col2:
        # Create a district heat map
        all_districts = sorted(set([district for sublist in df['districts_list'] for district in sublist]))
        program_types = sorted(df['program_type'].unique())
        
        # Create a matrix of counts
        district_program_matrix = np.zeros((len(all_districts), len(program_types)))
        
        for idx, district in enumerate(all_districts):
            for jdx, prog_type in enumerate(program_types):
                count = len(df[(df['districts'].str.contains(district)) & (df['program_type'] == prog_type)])
                district_program_matrix[idx, jdx] = count
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=district_program_matrix,
            x=program_types,
            y=all_districts,
            colorscale='Viridis',
            texttemplate="%{z}",
            textfont={"size": 10},
        ))
        fig_heatmap.update_layout(
            title="Districts by Program Type Heatmap",
            xaxis_title="Program Type",
            yaxis_title="School District",
            height=400
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

with tab2:
    # Create a treemap of district participation by program
    # First, we need to explode the districts_list to get one row per district per program
    exploded_df = filtered_df.explode('districts_list')
    
    treemap_data = exploded_df.groupby(['districts_list', 'program_type']).size().reset_index(name='count')
    
    fig_treemap = px.treemap(
        treemap_data,
        path=['districts_list', 'program_type'],
        values='count',
        title="District Participation by Program Type",
        color='program_type',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig_treemap.update_layout(
        margin=dict(t=50, l=25, r=25, b=25)
    )
    st.plotly_chart(fig_treemap, use_container_width=True)

# Performance Analysis
st.header("Program Performance Analysis")
col1, col2 = st.columns(2)

with col1:
    # Job placement rate by program type
    placement_by_type = filtered_df.groupby('program_type')['job_placement_rate'].mean().reset_index()
    placement_by_type['job_placement_rate'] = placement_by_type['job_placement_rate'] * 100
    
    fig_placement = px.bar(
        placement_by_type,
        x='program_type',
        y='job_placement_rate',
        title="Average Job Placement Rate by Program Type",
        color='job_placement_rate',
        color_continuous_scale=px.colors.sequential.Viridis,
        text=placement_by_type['job_placement_rate'].round(1).astype(str) + '%'
    )
    fig_placement.update_layout(
        xaxis_title="Program Type",
        yaxis_title="Job Placement Rate (%)",
        yaxis=dict(range=[0, 100])
    )
    fig_placement.update_traces(textposition='outside')
    st.plotly_chart(fig_placement, use_container_width=True)

with col2:
    # Scatter plot: enrollment vs. placement rate
    fig_scatter = px.scatter(
        filtered_df,
        x='total_students',
        y='job_placement_rate',
        color='program_type',
        size='total_students',
        hover_name='program_name',
        title="Enrollment vs. Job Placement Rate",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig_scatter.update_layout(
        xaxis_title="Total Enrollment",
        yaxis_title="Job Placement Rate",
        yaxis=dict(tickformat='.0%')
    )
    # Add a trend line
    fig_scatter.add_shape(
        type='line',
        x0=min(filtered_df['total_students']),
        y0=np.polyval(np.polyfit(filtered_df['total_students'], filtered_df['job_placement_rate'], 1),
                     min(filtered_df['total_students'])),
        x1=max(filtered_df['total_students']),
        y1=np.polyval(np.polyfit(filtered_df['total_students'], filtered_df['job_placement_rate'], 1),
                     max(filtered_df['total_students'])),
        line=dict(color='rgba(0,0,0,0.5)', width=2, dash='dash')
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# Instructor Analysis
st.header("Instructor Analysis")
col1, col2 = st.columns(2)

with col1:
    # Count of programs by instructor
    instructor_counts = filtered_df['instructor'].value_counts().reset_index()
    instructor_counts.columns = ['Instructor', 'Program Count']
    
    # Add total students
    instructor_students = filtered_df.groupby('instructor')['total_students'].sum().reset_index()
    instructor_students.columns = ['Instructor', 'Total Students']
    
    # Merge the data
    instructor_data = pd.merge(instructor_counts, instructor_students, on='Instructor')
    
    fig_instructors = px.bar(
        instructor_data,
        x='Instructor',
        y=['Program Count', 'Total Students'],
        title="Programs and Students by Instructor",
        barmode='group'
    )
    fig_instructors.update_layout(
        xaxis_title="Instructor",
        yaxis_title="Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_instructors, use_container_width=True)

with col2:
    # Analyze credentials impact on placement
    credentials_impact = filtered_df.groupby(['credentials', 'program_type'])['job_placement_rate'].mean().reset_index()
    credentials_impact['job_placement_rate'] = credentials_impact['job_placement_rate'] * 100
    
    fig_credentials = px.bar(
        credentials_impact,
        x='program_type',
        y='job_placement_rate',
        color='credentials',
        barmode='group',
        title="Impact of Instructor Credentials on Job Placement",
        text=credentials_impact['job_placement_rate'].round(1).astype(str) + '%',
        color_discrete_map={'Yes': '#2ecc71', 'No': '#e74c3c'}
    )
    fig_credentials.update_layout(
        xaxis_title="Program Type",
        yaxis_title="Job Placement Rate (%)",
        yaxis=dict(range=[0, 100]),
        legend_title="Credentialed Instructor",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_credentials.update_traces(textposition='outside')
    st.plotly_chart(fig_credentials, use_container_width=True)

# Certification Analysis
st.header("Certification Analysis")

# Extract and count certifications
all_certs = []
for cert_list in filtered_df['certifications'].str.split(', '):
    if cert_list[0] != 'None':
        all_certs.extend(cert_list)

cert_counts = pd.Series(all_certs).value_counts().reset_index()
cert_counts.columns = ['Certification', 'Count']
cert_counts = cert_counts.sort_values('Count', ascending=False)

fig_certs = px.bar(
    cert_counts,
    x='Certification',
    y='Count',
    title="Most Common Certifications Offered",
    color='Count',
    color_continuous_scale=px.colors.sequential.Viridis
)
fig_certs.update_layout(
    xaxis_title="Certification",
    yaxis_title="Number of Programs",
    xaxis={'categoryorder': 'total descending'}
)
st.plotly_chart(fig_certs, use_container_width=True)

# Correlation Analysis
st.header("Correlation Analysis")

# Calculate correlations
correlation_features = ['total_students', 'boys', 'girls', 'gender_ratio', 'credentials_binary', 'job_placement_rate']
corr_matrix = filtered_df[correlation_features].corr()

fig_corr = px.imshow(
    corr_matrix,
    text_auto=True,
    color_continuous_scale='RdBu_r',
    title="Correlation Between Program Features",
    zmin=-1,
    zmax=1
)
fig_corr.update_layout(
    height=500
)
st.plotly_chart(fig_corr, use_container_width=True)

# Program Data Table
st.header("Program Data Table")
st.dataframe(
    filtered_df[['program_name', 'program_type', 'total_students', 'boys', 'girls', 
                'instructor', 'credentials', 'job_placement_rate', 'certifications']],
    use_container_width=True
)

# Add a download button
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    "Download Filtered Data as CSV",
    csv,
    "mccc_filtered_data.csv",
    "text/csv",
    key='download-csv'
)

# Footer
st.markdown("""
---
*This dashboard is a demonstration of Streamlit capabilities using synthetic data based on Medina County Career Center programs.*
""")