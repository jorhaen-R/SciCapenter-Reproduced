function addRow(tableId, rowData) {
    const table = document.getElementById(tableId);
    if (!table) {
        console.error(`Table with id "${tableId}" not found.`);
        return;
    } 
    const newRow = table.insertRow();
    rowData.forEach(data => {
        const newCell = newRow.insertCell();
        newCell.textContent = data;
    });
    return newRow;
}
function deleteRow(tableId, rowIndex) {
    const table = document.getElementById(tableId);
    if (!table) {
        console.error(`Table with id "${tableId}" not found.`);
        return;
    }
    if (rowIndex < 0 || rowIndex >= table.rows.length) {
        console.error(`Row index "${rowIndex}" is out of bounds.`);
        return;
    }
    table.deleteRow(rowIndex);
}
function updateCell(tableId, rowIndex, cellIndex, newValue) {
    const table = document.getElementById(tableId);
    if (!table) {
        console.error(`Table with id "${tableId}" not found.`);
        return;
    } 
    if (rowIndex < 0 || rowIndex >= table.rows.length) {
        console.error(`Row index "${rowIndex}" is out of bounds.`);
        return;
    }   
    const row = table.rows[rowIndex];
    if (cellIndex < 0 || cellIndex >= row.cells.length) {
        console.error(`Cell index "${cellIndex}" is out of bounds.`);
        return;
    }
    row.cells[cellIndex].textContent = newValue;
}
function getCellValue(tableId, rowIndex, cellIndex) {
    const table = document.getElementById(tableId);
    if (!table) {
        console.error(`Table with id "${tableId}" not found.`);
        return null;
    }
    if (rowIndex < 0 || rowIndex >= table.rows.length) {
        console.error(`Row index "${rowIndex}" is out of bounds.`);
        return null;
    }
    const row = table.rows[rowIndex];
    if (cellIndex < 0 || cellIndex >= row.cells.length) {
        console.error(`Cell index "${cellIndex}" is out of bounds.`);
        return null;
    }
    return row.cells[cellIndex].textContent;
} 
export { addRow, deleteRow, updateCell, getCellValue };

