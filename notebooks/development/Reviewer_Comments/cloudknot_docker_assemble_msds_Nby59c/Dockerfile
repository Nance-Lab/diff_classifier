###############################################################################
# Dockerfile to build assemble_msds application container
# Based on arokem/python3-fiji:0.3
###############################################################################

# Use official python base image
FROM arokem/python3-fiji:0.3

# Install python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir git+https://github.com/ccurtis7/diff_classifier.git@Chad

# Create a default user. Available via runtime flag `--user cloudknot-user`.
# Add user to "staff" group.
# Give user a home directory.
RUN (id -u cloudknot-user >/dev/null 2>&1 || useradd cloudknot-user) \
    && addgroup cloudknot-user staff \
    && mkdir -p /home/cloudknot-user \
    && chown -R cloudknot-user:staff /home/cloudknot-user

ENV HOME /home/cloudknot-user

# Copy the python script
COPY assemble_msds.py /home/cloudknot-user/

# Set working directory
WORKDIR /home/cloudknot-user

# Set entrypoint
ENTRYPOINT ["python", "/home/cloudknot-user/assemble_msds.py"]
